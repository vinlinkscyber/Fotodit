//! Per-user protected local storage for AI provider keys.
//!
//! The user types a key into Fotodit on their own machine; it is sealed with the operating
//! system's per-user facility and written to one small file. Nothing is embedded in the app, sent
//! to chat, or exported: the API can report whether a key is saved ([`KeyStore::has`]), save,
//! replace or remove one, and hand the key to the in-process code that calls the provider
//! ([`KeyStore::get`]). There is deliberately no export or "show saved key" function.
//!
//! # What the protection is, and is not
//!
//! * Windows: DPAPI with the current user's key (see `windows.rs`). The file is useless on another
//!   machine or under another Windows account. Software running as the same Windows user can still
//!   unseal it, as can anyone who knows that user's Windows logon. It is protection at rest, not
//!   a defence against malware already running as the user.
//! * Other platforms: [`PlatformSealer`] refuses (`Unsupported`). No plaintext fallback is written.
//! * No cryptography is implemented here and no master secret is embedded: sealing is delegated
//!   to the OS.
//!
//! The in-memory [`SecretString`] hides its value from `Debug`/`Display` and overwrites its buffer
//! on drop on a best-effort basis (copies the OS or the HTTP layer make are outside its control).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
pub use windows::DpapiSealer;

/// Sealer used by [`default_store`]: DPAPI on Windows, a refusing placeholder elsewhere.
#[cfg(target_os = "windows")]
pub type PlatformSealer = DpapiSealer;
#[cfg(not(target_os = "windows"))]
pub type PlatformSealer = UnsupportedSealer;

const FILE_VERSION: u32 = 1;
const MAX_SECRET_BYTES: usize = 4096;

/// Why a key store operation failed. Messages never contain secret values.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KeyStoreError {
    #[error("protected key storage is not available on this platform")]
    Unsupported,
    #[error("the key is not acceptable: {0}")]
    InvalidSecret(&'static str),
    #[error("key file error: {0}")]
    Io(String),
    #[error("the saved key file is damaged or was written by a different protection method")]
    Corrupt,
    #[error("could not protect the key: {0}")]
    Seal(String),
    #[error("could not unlock the saved key (it may belong to another Windows user or machine): {0}")]
    Unseal(String),
}

/// Seals bytes with an OS facility. `context` binds a sealed value to its slot; it is not secret.
pub trait Sealer {
    fn name(&self) -> &'static str;
    fn seal(&self, plain: &[u8], context: &[u8]) -> Result<Vec<u8>, KeyStoreError>;
    fn unseal(&self, sealed: &[u8], context: &[u8]) -> Result<Vec<u8>, KeyStoreError>;
}

/// Placeholder for platforms without a supported protected store: refuses to save anything.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnsupportedSealer;

impl Sealer for UnsupportedSealer {
    fn name(&self) -> &'static str {
        "unsupported"
    }
    fn seal(&self, _: &[u8], _: &[u8]) -> Result<Vec<u8>, KeyStoreError> {
        Err(KeyStoreError::Unsupported)
    }
    fn unseal(&self, _: &[u8], _: &[u8]) -> Result<Vec<u8>, KeyStoreError> {
        Err(KeyStoreError::Unsupported)
    }
}

/// Overwrites a typed secret in place before it is released. Same limits as [`SecretString`].
pub fn wipe(text: &mut String) {
    text.zeroize();
}

/// A secret value that cannot be printed by accident.
pub struct SecretString(Vec<u8>);

impl SecretString {
    fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
    /// The value, for the in-process call that needs it (an HTTP header, for example).
    pub fn expose(&self) -> &str {
        std::str::from_utf8(&self.0).unwrap_or("")
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString(<redacted>)")
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        // Best effort: zeroize uses volatile stores, so the compiler cannot drop the wipe. It covers this buffer
        // only, not copies made elsewhere (the caller's request text, the network stack, a paste buffer).
        self.0.zeroize();
    }
}

#[derive(Default, Serialize, Deserialize)]
struct FileData {
    version: u32,
    sealer: String,
    entries: BTreeMap<String, String>,
    /// Slots the user removed on purpose. Callers must not fall back to any other credential source for them.
    #[serde(default)]
    disabled: BTreeSet<String>,
    /// Slots where the user chose to use the environment variable / `.env` file instead of a saved key.
    #[serde(default)]
    external: BTreeSet<String>,
}

/// Where a provider's credential comes from, as the user chose it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A key saved in this store. It is the only credential used for the provider.
    Saved,
    /// The user removed the key. No other source (environment, files, built-in) may stand in.
    Disabled,
    /// The user chose to use the provider's environment variable or `.env` file instead of a saved key.
    External,
    /// The user has not chosen. No key from any source is used until they do.
    Unconfigured,
}

/// A key file plus the sealer that protects its entries. `slot` names are short lowercase ids
/// such as `gemini`.
pub struct KeyStore<S: Sealer> {
    path: PathBuf,
    sealer: S,
}

fn valid_slot(slot: &str) -> bool {
    !slot.is_empty() && slot.len() <= 32 && slot.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

fn check_secret(secret: &str) -> Result<&str, KeyStoreError> {
    let s = secret.trim();
    if s.is_empty() {
        return Err(KeyStoreError::InvalidSecret("empty"));
    }
    if s.len() > MAX_SECRET_BYTES {
        return Err(KeyStoreError::InvalidSecret("too long"));
    }
    if s.chars().any(char::is_control) {
        return Err(KeyStoreError::InvalidSecret("contains control characters"));
    }
    Ok(s)
}

impl<S: Sealer> KeyStore<S> {
    pub fn new(path: impl Into<PathBuf>, sealer: S) -> Self {
        Self { path: path.into(), sealer }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read(&self) -> Result<FileData, KeyStoreError> {
        match std::fs::read(&self.path) {
            Ok(bytes) => {
                let data: FileData = serde_json::from_slice(&bytes).map_err(|_| KeyStoreError::Corrupt)?;
                if data.version != FILE_VERSION || data.sealer != self.sealer.name() {
                    return Err(KeyStoreError::Corrupt);
                }
                Ok(data)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Ok(FileData { version: FILE_VERSION, sealer: self.sealer.name().to_string(), entries: BTreeMap::new(), disabled: BTreeSet::new(), external: BTreeSet::new() })
            }
            Err(e) => Err(KeyStoreError::Io(e.to_string())),
        }
    }

    fn write(&self, data: &FileData) -> Result<(), KeyStoreError> {
        if data.entries.is_empty() && data.disabled.is_empty() && data.external.is_empty() {
            return match std::fs::remove_file(&self.path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(KeyStoreError::Io(e.to_string())),
            };
        }
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| KeyStoreError::Io(e.to_string()))?;
        }
        let json = serde_json::to_vec_pretty(data).map_err(|e| KeyStoreError::Io(e.to_string()))?;
        let tmp = self.path.with_extension("tmp");
        std::fs::write(&tmp, json).map_err(|e| KeyStoreError::Io(e.to_string()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
        }
        std::fs::rename(&tmp, &self.path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            KeyStoreError::Io(e.to_string())
        })
    }

    /// Whether a key is saved in `slot`. Does not unlock it.
    pub fn has(&self, slot: &str) -> Result<bool, KeyStoreError> {
        if !valid_slot(slot) {
            return Ok(false);
        }
        Ok(self.read()?.entries.contains_key(slot))
    }

    /// Saves or replaces the key in `slot`.
    pub fn set(&self, slot: &str, secret: &str) -> Result<(), KeyStoreError> {
        if !valid_slot(slot) {
            return Err(KeyStoreError::InvalidSecret("unknown provider slot"));
        }
        let secret = check_secret(secret)?;
        let sealed = self.sealer.seal(secret.as_bytes(), slot.as_bytes())?;
        let mut data = self.read()?;
        data.entries.insert(slot.to_string(), B64.encode(sealed));
        data.disabled.remove(slot);
        data.external.remove(slot);
        self.write(&data)
    }

    /// Removes the key in `slot` and records that the user did so, so no other source stands in for it
    /// (see [`Source::Disabled`]). Removing a missing key is not an error.
    pub fn remove(&self, slot: &str) -> Result<(), KeyStoreError> {
        if !valid_slot(slot) {
            return Err(KeyStoreError::InvalidSecret("unknown provider slot"));
        }
        let mut data = self.read()?;
        data.entries.remove(slot);
        data.disabled.insert(slot.to_string());
        data.external.remove(slot);
        self.write(&data)
    }

    /// Forgets the user's choice for `slot`: the key (if any) is deleted and the provider goes back to
    /// [`Source::Unconfigured`] (no key is used). An explicit user action.
    pub fn reset(&self, slot: &str) -> Result<(), KeyStoreError> {
        let mut data = self.read()?;
        let changed = data.entries.remove(slot).is_some() | data.disabled.remove(slot) | data.external.remove(slot);
        if changed {
            self.write(&data)?;
        }
        Ok(())
    }

    /// Records the user's choice to use the environment variable / `.env` file for `slot`. Refused while a key is
    /// saved there: remove it first, so a saved key is never deleted as a side effect.
    pub fn use_external(&self, slot: &str) -> Result<(), KeyStoreError> {
        if !valid_slot(slot) {
            return Err(KeyStoreError::InvalidSecret("unknown provider slot"));
        }
        let mut data = self.read()?;
        if data.entries.contains_key(slot) {
            return Err(KeyStoreError::InvalidSecret("a key is saved here; remove it first"));
        }
        data.disabled.remove(slot);
        data.external.insert(slot.to_string());
        self.write(&data)
    }

    /// The credential source the user chose for `slot`.
    pub fn source(&self, slot: &str) -> Result<Source, KeyStoreError> {
        let data = self.read()?;
        Ok(if data.entries.contains_key(slot) {
            Source::Saved
        } else if data.disabled.contains(slot) {
            Source::Disabled
        } else if data.external.contains(slot) {
            Source::External
        } else {
            Source::Unconfigured
        })
    }

    /// The saved key, for the in-process code that calls the provider. `Ok(None)` when none is saved.
    pub fn get(&self, slot: &str) -> Result<Option<SecretString>, KeyStoreError> {
        let data = self.read()?;
        let Some(b64) = data.entries.get(slot) else {
            return Ok(None);
        };
        let sealed = B64.decode(b64).map_err(|_| KeyStoreError::Corrupt)?;
        let plain = self.sealer.unseal(&sealed, slot.as_bytes())?;
        Ok(Some(SecretString::new(plain)))
    }
}

/// An AI provider that can have a saved key. `slot` is the storage id, `env_var` the legacy
/// environment variable the app also reads, shown to the user as a credential source.
#[derive(Clone, Copy, Debug)]
pub struct Provider {
    pub slot: &'static str,
    pub label: &'static str,
    pub env_var: &'static str,
}

/// Providers listed on the AI providers screen. `fotodit-automation` maps its provider ids onto these
/// slots (checked by a test there).
pub const PROVIDERS: [Provider; 5] = [
    Provider { slot: "gemini", label: "Google Gemini", env_var: "GEMINI_API_KEY" },
    Provider { slot: "openrouter", label: "OpenRouter", env_var: "OPENROUTER_API_KEY" },
    Provider { slot: "cohere", label: "Cohere", env_var: "COHERE_API_KEY" },
    Provider { slot: "pollinations", label: "Pollinations", env_var: "POLLINATIONS_API_KEY" },
    Provider { slot: "lajiwa", label: "Lajiwa", env_var: "LAJIWA_API_KEY" },
];

/// Default location: `%APPDATA%\Fotodit\ai_providers.v1.json` (Windows). `None` when the platform
/// has no per-user application data directory we use.
pub fn default_path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA").filter(|v| !v.is_empty())?;
    Some(PathBuf::from(base).join("Fotodit").join("ai_providers.v1.json"))
}

/// Whether this platform can protect saved keys. When `false`, saving is refused rather than
/// storing anything unprotected.
pub const fn supported() -> bool {
    cfg!(windows)
}

/// The store the app uses, or `None` when there is no per-user data directory.
pub fn default_store() -> Option<KeyStore<PlatformSealer>> {
    Some(KeyStore::new(default_path()?, PlatformSealer::default()))
}

/// A reversible, non-cryptographic stand-in for tests, so no real credentials or OS store is used.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use super::{KeyStoreError, Sealer};

    #[derive(Clone, Copy, Debug, Default)]
    pub struct MockSealer;

    impl Sealer for MockSealer {
        fn name(&self) -> &'static str {
            "mock"
        }
        fn seal(&self, plain: &[u8], context: &[u8]) -> Result<Vec<u8>, KeyStoreError> {
            let mut v = b"MOCK:".to_vec();
            v.extend_from_slice(context);
            v.push(b':');
            v.extend(plain.iter().rev());
            Ok(v)
        }
        fn unseal(&self, sealed: &[u8], context: &[u8]) -> Result<Vec<u8>, KeyStoreError> {
            let mut prefix = b"MOCK:".to_vec();
            prefix.extend_from_slice(context);
            prefix.push(b':');
            let body = sealed.strip_prefix(prefix.as_slice()).ok_or_else(|| KeyStoreError::Unseal("wrong slot or foreign data".into()))?;
            Ok(body.iter().rev().copied().collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::MockSealer;
    use super::*;

    fn store(name: &str) -> (KeyStore<MockSealer>, PathBuf) {
        let dir = std::env::temp_dir().join(format!("fotodit-keystore-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("keys.json");
        (KeyStore::new(&path, MockSealer), dir)
    }

    #[test]
    fn set_after_remove_makes_it_saved_again() {
        let (s, dir) = store("setafter");
        s.set("gemini", "k1").unwrap();
        assert_eq!(s.source("gemini").unwrap(), Source::Saved);
        s.remove("gemini").unwrap();
        s.set("gemini", "k2").unwrap();
        assert_eq!(s.source("gemini").unwrap(), Source::Saved);
        assert_eq!(s.get("gemini").unwrap().unwrap().expose(), "k2");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn external_is_an_explicit_choice_and_never_deletes_a_saved_key() {
        let (s, dir) = store("ext");
        assert_eq!(s.source("cohere").unwrap(), Source::Unconfigured);
        s.use_external("cohere").unwrap();
        assert_eq!(s.source("cohere").unwrap(), Source::External);
        s.set("cohere", "key-1234").unwrap();
        assert_eq!(s.source("cohere").unwrap(), Source::Saved);
        assert!(s.use_external("cohere").is_err());
        assert_eq!(s.get("cohere").unwrap().unwrap().expose(), "key-1234");
        s.remove("cohere").unwrap();
        assert_eq!(s.source("cohere").unwrap(), Source::Disabled);
        s.use_external("cohere").unwrap();
        assert_eq!(s.source("cohere").unwrap(), Source::External);
        s.reset("cohere").unwrap();
        assert_eq!(s.source("cohere").unwrap(), Source::Unconfigured);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn save_replace_remove_round_trip() {
        let (s, dir) = store("rt");
        assert_eq!(s.has("gemini"), Ok(false));
        s.set("gemini", "  test-key-AAA  ").unwrap();
        assert_eq!(s.has("gemini"), Ok(true));
        assert_eq!(s.get("gemini").unwrap().unwrap().expose(), "test-key-AAA");
        s.set("gemini", "test-key-BBB").unwrap();
        assert_eq!(s.get("gemini").unwrap().unwrap().expose(), "test-key-BBB");
        s.remove("gemini").unwrap();
        assert_eq!(s.has("gemini"), Ok(false));
        assert!(s.get("gemini").unwrap().is_none());
        assert_eq!(s.source("gemini").unwrap(), Source::Disabled, "remove is remembered, not forgotten");
        s.reset("gemini").unwrap();
        assert_eq!(s.source("gemini").unwrap(), Source::Unconfigured);
        assert!(!s.path().exists(), "the file goes away when nothing is saved or disabled");
        s.remove("gemini").unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_file_never_contains_the_plain_value() {
        let (s, dir) = store("plain");
        s.set("cohere", "plain-value-12345").unwrap();
        let text = std::fs::read_to_string(s.path()).unwrap();
        assert!(!text.contains("plain-value-12345"));
        assert!(text.contains("\"sealer\": \"mock\""));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_bad_values_and_slots() {
        let (s, dir) = store("bad");
        assert_eq!(s.set("gemini", "   "), Err(KeyStoreError::InvalidSecret("empty")));
        assert_eq!(s.set("gemini", "a\nb"), Err(KeyStoreError::InvalidSecret("contains control characters")));
        assert_eq!(s.set("gemini", &"x".repeat(MAX_SECRET_BYTES + 1)), Err(KeyStoreError::InvalidSecret("too long")));
        assert!(s.set("../evil", "k").is_err());
        assert!(!s.path().exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_sealed_value_cannot_be_moved_to_another_slot() {
        let (s, dir) = store("swap");
        s.set("gemini", "key-one").unwrap();
        let mut data = s.read().unwrap();
        let moved = data.entries["gemini"].clone();
        data.entries.insert("cohere".into(), moved);
        s.write(&data).unwrap();
        assert!(matches!(s.get("cohere"), Err(KeyStoreError::Unseal(_))));
        assert_eq!(s.get("gemini").unwrap().unwrap().expose(), "key-one");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn damaged_or_foreign_files_are_reported_not_overwritten() {
        let (s, dir) = store("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(s.path(), b"not json").unwrap();
        assert_eq!(s.has("gemini"), Err(KeyStoreError::Corrupt));
        assert_eq!(s.set("gemini", "k-1234"), Err(KeyStoreError::Corrupt));
        assert_eq!(std::fs::read(s.path()).unwrap(), b"not json");
        std::fs::write(s.path(), br#"{"version":1,"sealer":"dpapi-current-user","entries":{}}"#).unwrap();
        assert_eq!(s.has("gemini"), Err(KeyStoreError::Corrupt), "another sealer's file is not read with this one");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unsupported_platforms_refuse_and_write_nothing() {
        let dir = std::env::temp_dir().join(format!("fotodit-keystore-unsup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = KeyStore::new(dir.join("keys.json"), UnsupportedSealer);
        assert_eq!(s.set("gemini", "key-1234"), Err(KeyStoreError::Unsupported));
        assert!(!dir.join("keys.json").exists());
    }

    #[test]
    fn wipe_empties_a_typed_secret() {
        let mut typed = String::from("typed-key-123");
        wipe(&mut typed);
        assert!(typed.is_empty());
    }

    #[test]
    fn debug_hides_the_value() {
        let (s, dir) = store("dbg");
        s.set("gemini", "super-secret-9").unwrap();
        let v = s.get("gemini").unwrap().unwrap();
        assert_eq!(format!("{v:?}"), "SecretString(<redacted>)");
        let _ = std::fs::remove_dir_all(dir);
    }
}
