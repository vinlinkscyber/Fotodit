//! Windows DPAPI (`CryptProtectData` / `CryptUnprotectData`), current-user scope.
//!
//! This is the only file in the crate that allows `unsafe`: the DPAPI functions take and return
//! raw `CRYPT_INTEGER_BLOB` pointers. Each block has a `SAFETY:` note.
#![allow(unsafe_code)]

use windows::Win32::Foundation::{HLOCAL, LocalFree};
use windows::Win32::Security::Cryptography::{CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptProtectData, CryptUnprotectData};
use windows::core::w;

use crate::{KeyStoreError, Sealer};

/// Seals with the logged-in Windows user's DPAPI master key (no `CRYPTPROTECT_LOCAL_MACHINE`).
///
/// What that protects: the blob on disk cannot be decrypted on another machine or by another
/// Windows user account. What it does not: any program running as the same user can call
/// `CryptUnprotectData` on the blob, and so can anyone who has that user's logon (Windows
/// derives the master key from it). The `context` argument goes in as DPAPI "optional entropy":
/// it binds each blob to its provider so blobs cannot be swapped between slots. It is not a secret.
#[derive(Clone, Copy, Debug, Default)]
pub struct DpapiSealer;

fn blob(bytes: &[u8]) -> Result<CRYPT_INTEGER_BLOB, KeyStoreError> {
    let len = u32::try_from(bytes.len()).map_err(|_| KeyStoreError::Seal("value too large".into()))?;
    // The API never writes through `pbData` of an input blob; the cast only satisfies its type.
    Ok(CRYPT_INTEGER_BLOB { cbData: len, pbData: bytes.as_ptr().cast_mut() })
}

/// Copies a DPAPI-allocated output blob and frees it. With `wipe`, the allocation is zeroed first, because for
/// an unseal result it holds the plaintext key.
fn take(out: CRYPT_INTEGER_BLOB, wipe: bool) -> Vec<u8> {
    if out.pbData.is_null() {
        return Vec::new();
    }
    // SAFETY: on success DPAPI returns `cbData` readable bytes at `pbData`, allocated with
    // LocalAlloc; we copy them and free the allocation exactly once.
    unsafe {
        let copy = std::slice::from_raw_parts(out.pbData, out.cbData as usize).to_vec();
        if wipe {
            // Volatile stores so the compiler cannot drop them (the stand-in for SecureZeroMemory).
            for i in 0..out.cbData as usize {
                std::ptr::write_volatile(out.pbData.add(i), 0);
            }
        }
        let _ = LocalFree(Some(HLOCAL(out.pbData.cast())));
        copy
    }
}

impl Sealer for DpapiSealer {
    fn name(&self) -> &'static str {
        "dpapi-current-user"
    }

    fn seal(&self, plain: &[u8], context: &[u8]) -> Result<Vec<u8>, KeyStoreError> {
        let input = blob(plain)?;
        let entropy = blob(context)?;
        let mut out = CRYPT_INTEGER_BLOB::default();
        // SAFETY: `input`, `entropy` and `out` are valid for the call; the input buffers outlive
        // it; `out` is only read by `take` after success.
        unsafe { CryptProtectData(&raw const input, w!("Fotodit AI provider key"), Some(&raw const entropy), None, None, CRYPTPROTECT_UI_FORBIDDEN, &raw mut out) }
            .map_err(|e| KeyStoreError::Seal(format!("CryptProtectData failed: {e}")))?;
        Ok(take(out, false))
    }

    fn unseal(&self, sealed: &[u8], context: &[u8]) -> Result<Vec<u8>, KeyStoreError> {
        let input = blob(sealed)?;
        let entropy = blob(context)?;
        let mut out = CRYPT_INTEGER_BLOB::default();
        // SAFETY: as in `seal`; no description out-pointer is requested.
        unsafe { CryptUnprotectData(&raw const input, None, Some(&raw const entropy), None, None, CRYPTPROTECT_UI_FORBIDDEN, &raw mut out) }
            .map_err(|e| KeyStoreError::Unseal(format!("CryptUnprotectData failed: {e}")))?;
        Ok(take(out, true))
    }
}
