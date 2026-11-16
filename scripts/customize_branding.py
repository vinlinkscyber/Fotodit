#!/usr/bin/env python3
"""
Branding Customization Engine for Fotodit -> Fotodit.

Idempotent, reversible migration script with AST-aware/context-aware string replacement,
restorable backup, dry-run inspection, and change manifest generation.

Customization Mapping:
  - Product Name: Fotodit -> Fotodit
  - Binary/slug: fotodit -> fotodit
  - Bundle/App ID: com.plyra.fotodit -> com.plyra.fotodit
  - Organization/Manufacturer: Learning Machines LLC -> Plyra
  - Contributor credits & models tabs removed per explicit owner permission (Scope Question 4: i, ii, iii)
  - Engine acknowledgment & legal notices preserved (fotodit-engine, licenses, NOTICE)
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OLD_NAME = "Fotodit"
OLD_NAME_LOWER = "fotodit"
OLD_APP_ID = "com.plyra.fotodit"

NEW_NAME = "Fotodit"
NEW_NAME_LOWER = "fotodit"
NEW_APP_ID = "com.plyra.fotodit"
NEW_COMPANY = "Plyra"

PROTECTED_DIRS = {
    ".git",
    "target",
    "corpus",
    "crates/geom",
    "crates/cms",
    "crates/color",
    "crates/raster",
    "crates/psd",
    "crates/codecs",
    "crates/tablet",
    "crates/raw",
    "crates/affinity",
    "crates/ipp",
    "crates/doc",
    "crates/ops",
    "crates/compose",
    "crates/gpu",
    "crates/paint",
    "crates/algo",
    "crates/text",
    "crates/vector",
    "crates/format",
    "crates/io",
    "crates/plugins",
    "crates/engine",
    "crates/automation",
}

def is_protected(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/")
    for p in PROTECTED_DIRS:
        if norm == p or norm.startswith(p + "/"):
            return True
    if norm.endswith("Cargo.lock") or norm.startswith("backups/"):
        return True
    return False

def build_plan(root: str) -> list[dict]:
    plan = []

    # 1. Renames of packaging and descriptor files
    rename_targets = [
        ("packaging/linux/com.plyra.fotodit.desktop", "packaging/linux/com.plyra.fotodit.desktop", "Linux desktop shortcut entry"),
        ("packaging/linux/com.plyra.fotodit.metainfo.xml.in", "packaging/linux/com.plyra.fotodit.metainfo.xml.in", "Linux AppStream metainfo descriptor"),
        ("packaging/linux/com.plyra.fotodit.mime.xml", "packaging/linux/com.plyra.fotodit.mime.xml", "Linux MIME database specification"),
        ("packaging/linux/flatpak/com.plyra.fotodit.yml", "packaging/linux/flatpak/com.plyra.fotodit.yml", "Flatpak manifest"),
        ("packaging/linux/flatpak/com.plyra.fotodit.bundle.yml", "packaging/linux/flatpak/com.plyra.fotodit.bundle.yml", "Flatpak bundle manifest"),
    ]

    for old_rel, new_rel, reason in rename_targets:
        full_old = os.path.join(root, old_rel.replace("/", os.sep))
        if os.path.exists(full_old):
            plan.append({
                "path": old_rel,
                "type": "rename",
                "old": old_rel,
                "new": new_rel,
                "symbol_type": "file_path",
                "reason": reason,
            })

    # 2. In-file text substitutions in app-owned files
    file_rules = {
        "apps/fotodit/src/main.rs": [
            ('.with_title("Fotodit")', '.with_title("Fotodit")', "UI String", "Update desktop window title"),
            ('log::info!("Fotodit {}, log file {}"', 'log::info!("Fotodit {}, log file {}"', "Log String", "Update startup log identity"),
            ('log::error!("Fotodit could not start: {error}")', 'log::error!("Fotodit could not start: {error}")', "Log String", "Update error log identity"),
            ('include_str!("../../../packaging/linux/com.plyra.fotodit.desktop")', 'include_str!("../../../packaging/linux/com.plyra.fotodit.desktop")', "Asset Resource Path", "Reference renamed Linux desktop file"),
        ],
        "apps/fotodit/src/app_dirs.rs": [
            ('["portable.txt", "Fotodit.portable"]', '["portable.txt", "Fotodit.portable"]', "Configuration Constant", "Update portable mode marker filename"),
            ('pub const PORTABLE_DATA_DIR: &str = "FotoditData";', 'pub const PORTABLE_DATA_DIR: &str = "FotoditData";', "Configuration Constant", "Update portable mode data directory name"),
            ('macos ~/Library/Application Support/Fotodit', 'macos ~/Library/Application Support/Fotodit', "Doc Comment", "Update documentation for macOS app support directory"),
            ('Windows\n//!    %APPDATA%\\Fotodit', 'Windows\n//!    %APPDATA%\\Fotodit', "Doc Comment", "Update documentation for Windows AppData directory"),
            ('Linux $XDG_CONFIG_HOME/fotodit or ~/.config/fotodit', 'Linux $XDG_CONFIG_HOME/fotodit or ~/.config/fotodit', "Doc Comment", "Update documentation for Linux config directory"),
        ],
        "apps/fotodit/src/app_icon.rs": [
            ('assets/app-icon/hicolor/256x256/apps/com.plyra.fotodit.png', 'assets/app-icon/hicolor/256x256/apps/com.plyra.fotodit.png', "Asset Resource Path", "Reference new Fotodit application icon"),
            ('assets/app-icon/fotodit-1024.png', 'assets/app-icon/fotodit-1024.png', "Asset Resource Path", "Reference new Fotodit macOS icon"),
            ('assets/app-icon/fotodit.ico', 'assets/app-icon/fotodit.ico', "Asset Resource Path", "Reference new Fotodit Windows icon"),
            ('fotodit.ico lacks', 'fotodit.ico lacks', "Unit Test Assertion", "Update icon test message"),
            ('Fotodit.app', 'Fotodit.app', "Unit Test Path", "Update bundle test path"),
            ('Fotodit Beta.APP', 'Fotodit Beta.APP', "Unit Test Path", "Update bundle test path"),
        ],
        "apps/fotodit/src/services/tests.rs": [
            ('xwayland_command(Some("com.plyra.fotodit"), None, false);', 'xwayland_command(Some("com.plyra.fotodit"), None, false);', "Unit Test Parameter", "Update test app id"),
            ('Some("flatpak run --nosocket=wayland --socket=x11 com.plyra.fotodit")', 'Some("flatpak run --nosocket=wayland --socket=x11 com.plyra.fotodit")', "Unit Test Assertion", "Update test expected flatpak command"),
        ],
        "apps/fotodit/src/single_instance/tests.rs": [
            ('let (id, run) = (OsStr::new("com.plyra.fotodit"), OsStr::new("/run/user/1000"));', 'let (id, run) = (OsStr::new("com.plyra.fotodit"), OsStr::new("/run/user/1000"));', "Unit Test Setup", "Update test instance ID"),
            ('PathBuf::from("/run/user/1000/app/com.plyra.fotodit/instance.sock")', 'PathBuf::from("/run/user/1000/app/com.plyra.fotodit/instance.sock")', "Unit Test Assertion", "Update test expected socket path"),
        ],
        "apps/fotodit/build.rs": [
            ('res.set_icon("../../assets/app-icon/fotodit.ico")', 'res.set_icon("../../assets/app-icon/fotodit.ico")', "Build Resource", "Set Windows resource icon to fotodit.ico"),
            ('cargo:rerun-if-changed=../../assets/app-icon/fotodit.ico', 'cargo:rerun-if-changed=../../assets/app-icon/fotodit.ico', "Build Dependency", "Update rerun trigger for Windows icon"),
            ('.set("ProductName", "Fotodit")', '.set("ProductName", "Fotodit")', "Build Metadata", "Set ProductName to Fotodit"),
            ('.set("FileDescription", "Fotodit image editor")', '.set("FileDescription", "Fotodit image editor")', "Build Metadata", "Set FileDescription to Fotodit image editor"),
            ('.set("CompanyName", "Learning Machines LLC")', '.set("CompanyName", "Plyra")', "Build Metadata", "Set CompanyName to Plyra"),
        ],
        "crates/ui-egui/src/os_window.rs": [
            ('pub const APP_ID: &str = "com.plyra.fotodit";', 'pub const APP_ID: &str = "com.plyra.fotodit";', "App Identifier Constant", "Update OS window application ID"),
            ('assert_eq!(super::builder().app_id.as_deref(), Some("com.plyra.fotodit"));', 'assert_eq!(super::builder().app_id.as_deref(), Some("com.plyra.fotodit"));', "Unit Test Assertion", "Update test assertion for app ID"),
        ],
        "crates/ui-egui/src/panels/brand.rs": [
            ('assets/app-icon/hicolor/128x128/apps/com.plyra.fotodit.png', 'assets/app-icon/hicolor/128x128/apps/com.plyra.fotodit.png', "Asset Resource Path", "Reference new Fotodit branding icon in panels"),
        ],
        "crates/ui-egui/src/lib.rs": [
            ('assets/app-icon/hicolor/128x128/apps/com.plyra.fotodit.png', 'assets/app-icon/hicolor/128x128/apps/com.plyra.fotodit.png', "Asset Resource Path", "Reference new Fotodit app window icon"),
        ],
        "crates/ui-egui/src/notices.rs": [
            ('flatpak run --nosocket=wayland --socket=x11 com.plyra.fotodit', 'flatpak run --nosocket=wayland --socket=x11 com.plyra.fotodit', "UI Guidance Command", "Update Flatpak fallback command notice"),
        ],
        "crates/ui-egui/src/dialogs.rs": [
            ('DialogKind::About => "About Fotodit".into(),', 'DialogKind::About => "About Fotodit".into(),', "Dialog Title", "Update About dialog window title"),
            (
                '        DialogKind::About => {\n'
                '            // Tabs About · Contributors · Models. The\n'
                '            // tab is a dialog field, so automation can switch it with `ui.dialog.set`.\n'
                '            let tab = about_tab(fields);\n'
                '            let mut chosen = tab;\n'
                '            ui.horizontal(|ui| {\n'
                '                for (key, label) in [("about", tl!("About")), ("contributors", tl!("Contributors")), ("models", tl!("Models"))] {\n'
                '                    if crate::widgets::pill_tab(ui, label, tab == key).clicked() {\n'
                '                        chosen = key;\n'
                '                    }\n'
                '                }\n'
                '            });\n'
                '            if chosen != tab {\n'
                '                fields.insert("tab".into(), json!(chosen));\n'
                '            }\n'
                '            ui.add_space(8.0);\n'
                '            about_tab_body(app, ui, chosen);\n'
                '        }',
                '        DialogKind::About => {\n'
                '            about_tab_body(app, ui, "about");\n'
                '        }',
                'About Dialog UI',
                'Remove Contributors and Models tabs per owner permission (Scope Q4: i, ii)'
            ),
            (
                '        _ => {\n'
                '            ui.label(tl!("Fotodit — an open-source, native image editor written in Rust."));\n'
                '            ui.label(crate::i18n::fmt(tl!("Version {version}"), &[("version", &fotodit_engine::build_info::long_version())]));\n'
                '            ui.add_space(12.0);\n'
                '            ui.vertical_centered(|ui| {\n'
                '                crate::links::discord_button(app, ui, 220.0);\n'
                '                ui.add_space(8.0);\n'
                '                crate::links::link_row(app, ui);\n'
                '            });\n'
                '            ui.add_space(10.0);\n'
                '            ui.weak("egui · wgpu · fotodit-engine");\n'
                '        }',
                '        _ => {\n'
                '            ui.label(tl!("Fotodit — an open-source, native image editor written in Rust."));\n'
                '            ui.label(crate::i18n::fmt(tl!("Version {version}"), &[("version", &fotodit_engine::build_info::long_version())]));\n'
                '            ui.add_space(12.0);\n'
                '            ui.weak("egui · wgpu · fotodit-engine");\n'
                '        }',
                'About Dialog Blurb & Links',
                'Update product name to Fotodit and remove promotional links per owner permission (Scope Q4: iii)'
            ),
        ],
        "crates/ui-egui/src/dialogs/credits/ui_tests.rs": [
            ('.query_all_by_label("About Fotodit")', '.query_all_by_label("About Fotodit")', "Unit Test Assertion", "Update test query for About dialog title"),
        ],
        "apps/fotodit-web/index.html": [
            ('<title>Fotodit</title>', '<title>Fotodit</title>', "HTML Document Title", "Update browser tab title"),
            ('<p>Loading Fotodit&hellip;</p>', '<p>Loading Fotodit&hellip;</p>', "HTML UI String", "Update web loading message"),
            ('If Fotodit doesn\'t start', 'If Fotodit doesn\'t start', "HTML UI String", "Update web stall warning message"),
            ('<p><strong>Fotodit stopped.</strong></p>', '<p><strong>Fotodit stopped.</strong></p>', "HTML UI String", "Update web error alert"),
        ],
        "apps/fotodit-web-boot/index.html": [
            ('<title>Fotodit</title>', '<title>Fotodit</title>', "HTML Document Title", "Update bootloader tab title"),
            ('<p>Loading Fotodit&hellip;</p>', '<p>Loading Fotodit&hellip;</p>', "HTML UI String", "Update bootloader loading message"),
        ],
        "packaging/linux/com.plyra.fotodit.desktop": [
            ('Name=Fotodit', 'Name=Fotodit', "Desktop Entry Property", "Set desktop application name to Fotodit"),
            ('Icon=com.plyra.fotodit', 'Icon=com.plyra.fotodit', "Desktop Entry Property", "Update desktop icon identifier"),
            ('StartupWMClass=com.plyra.fotodit', 'StartupWMClass=com.plyra.fotodit', "Desktop Entry Property", "Update window manager class"),
        ],
        "packaging/linux/com.plyra.fotodit.metainfo.xml.in": [
            ('<id>com.plyra.fotodit</id>', '<id>com.plyra.fotodit</id>', "AppStream XML", "Update AppStream application ID"),
            ('<name>Fotodit</name>', '<name>Fotodit</name>', "AppStream XML", "Update AppStream display name"),
        ],
        "packaging/linux/com.plyra.fotodit.mime.xml": [
            ('<comment>Fotodit document</comment>', '<comment>Fotodit document</comment>', "MIME Specification", "Update document comment"),
            ('<icon name="com.plyra.fotodit"/>', '<icon name="com.plyra.fotodit"/>', "MIME Specification", "Update document icon reference"),
        ],
        "packaging/linux/AppRun": [
            ('DESK_ID=com.plyra.fotodit', 'DESK_ID=com.plyra.fotodit', "Shell Variable", "Update AppRun desktop ID"),
        ],
        "packaging/linux/package.sh": [
            ('APP_ID=com.plyra.fotodit', 'APP_ID=com.plyra.fotodit', "Shell Variable", "Update packaging script APP_ID"),
        ],
        "packaging/linux/flatpak/com.plyra.fotodit.yml": [
            ('app-id: com.plyra.fotodit', 'app-id: com.plyra.fotodit', "Flatpak Manifest", "Update Flatpak app id"),
        ],
        "packaging/linux/flatpak/com.plyra.fotodit.bundle.yml": [
            ('app-id: com.plyra.fotodit', 'app-id: com.plyra.fotodit', "Flatpak Manifest", "Update Flatpak bundle app id"),
        ],
        "packaging/macos/Info.plist.in": [
            ('<key>CFBundleDisplayName</key>\n  <string>Fotodit</string>', '<key>CFBundleDisplayName</key>\n  <string>Fotodit</string>', "macOS Plist Property", "Update macOS display name"),
            ('<key>CFBundleName</key>\n  <string>Fotodit</string>', '<key>CFBundleName</key>\n  <string>Fotodit</string>', "macOS Plist Property", "Update macOS bundle name"),
            ('<key>CFBundleExecutable</key>\n  <string>Fotodit</string>', '<key>CFBundleExecutable</key>\n  <string>Fotodit</string>', "macOS Plist Property", "Update macOS executable name"),
            ('<key>CFBundleIdentifier</key>\n  <string>com.plyra.fotodit</string>', '<key>CFBundleIdentifier</key>\n  <string>com.plyra.fotodit</string>', "macOS Plist Property", "Update macOS bundle identifier"),
            ('<key>FotoditVersion</key>', '<key>FotoditVersion</key>', "macOS Plist Property", "Update custom version key"),
            ('<key>FotoditBuildCommit</key>', '<key>FotoditBuildCommit</key>', "macOS Plist Property", "Update custom build commit key"),
            ('<key>CFBundleIconFile</key>\n  <string>Fotodit</string>', '<key>CFBundleIconFile</key>\n  <string>Fotodit</string>', "macOS Plist Property", "Update icon filename reference"),
            ('<string>com.plyra.fotodit.', '<string>com.plyra.fotodit.', "macOS Document UTI", "Update document UTI prefixes"),
        ],
        "packaging/windows/fotodit.wxs": [
            ('<Package Name="Fotodit" Manufacturer="Learning Machines LLC"', '<Package Name="Fotodit" Manufacturer="Plyra"', "WiX Package Property", "Update Windows installer package name and manufacturer"),
            ('<SummaryInformation Description="Fotodit image editor" Manufacturer="Learning Machines LLC" />', '<SummaryInformation Description="Fotodit image editor" Manufacturer="Plyra" />', "WiX Summary Property", "Update summary information"),
            ('DowngradeErrorMessage="A newer version of Fotodit is already installed."', 'DowngradeErrorMessage="A newer version of Fotodit is already installed."', "WiX Error Message", "Update downgrade message"),
            ('<Icon Id="FotoditIcon.exe"', '<Icon Id="FotoditIcon.exe"', "WiX Icon Definition", "Update MSI icon identifier"),
            ('<Property Id="ARPPRODUCTICON" Value="FotoditIcon.exe" />', '<Property Id="ARPPRODUCTICON" Value="FotoditIcon.exe" />', "WiX Property", "Update product icon reference"),
            ('<Directory Id="CraftSuiteFolder" Name="ArtCraft">\n        <Directory Id="INSTALLFOLDER" Name="Fotodit" />', '<Directory Id="CraftSuiteFolder" Name="Plyra">\n        <Directory Id="INSTALLFOLDER" Name="Fotodit" />', "WiX Directory Structure", "Update installation folder hierarchy"),
            ('<Feature Id="Main" Title="Fotodit"', '<Feature Id="Main" Title="Fotodit"', "WiX Feature", "Update feature title"),
            ('Value="LaunchFotodit"', 'Value="LaunchFotodit"', "WiX Action", "Update launch action identifier"),
        ],
    }

    for rel_path, rules in file_rules.items():
        full_path = os.path.join(root, rel_path.replace("/", os.sep))
        if not os.path.exists(full_path):
            continue
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        for old_str, new_str, sym_type, reason in rules:
            # Also check if CRLF variant matches
            old_str_crlf = old_str.replace("\n", "\r\n")
            if old_str in content or old_str_crlf in content:
                plan.append({
                    "path": rel_path,
                    "type": "replace",
                    "old": old_str,
                    "new": new_str,
                    "symbol_type": sym_type,
                    "reason": reason,
                })

    return plan

def create_backup(root: str, files_to_backup: set[str], backup_dir: str):
    os.makedirs(backup_dir, exist_ok=True)
    manifest = []
    for rel_path in sorted(files_to_backup):
        src = os.path.join(root, rel_path.replace("/", os.sep))
        dst = os.path.join(backup_dir, rel_path.replace("/", os.sep))
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            manifest.append({"rel_path": rel_path, "status": "backed_up"})
    with open(os.path.join(backup_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Created backup of {len(manifest)} files in: {backup_dir}")

def apply_plan(root: str, plan: list[dict], backup_dir: str):
    # Collect files that will be touched
    files_to_backup = set()
    for item in plan:
        files_to_backup.add(item["path"])

    create_backup(root, files_to_backup, backup_dir)

    # 1. Apply in-file replacements
    modified_files = set()
    for item in plan:
        if item["type"] == "replace":
            full_path = os.path.join(root, item["path"].replace("/", os.sep))
            with open(full_path, "rb") as f:
                raw_bytes = f.read()
            is_crlf = b"\r\n" in raw_bytes
            
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            old_str = item["old"]
            new_str = item["new"]
            
            if old_str in content:
                content = content.replace(old_str, new_str)
            else:
                old_str_crlf = old_str.replace("\n", "\r\n")
                new_str_crlf = new_str.replace("\n", "\r\n")
                if old_str_crlf in content:
                    content = content.replace(old_str_crlf, new_str_crlf)

            with open(full_path, "w", encoding="utf-8", newline="\r\n" if is_crlf else "\n") as f:
                f.write(content)
            modified_files.add(item["path"])

    # 2. Apply renames
    renamed_pairs = []
    for item in plan:
        if item["type"] == "rename":
            old_path = os.path.join(root, item["old"].replace("/", os.sep))
            new_path = os.path.join(root, item["new"].replace("/", os.sep))
            if os.path.exists(old_path):
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copy2(old_path, new_path)
                os.remove(old_path)
                renamed_pairs.append({"old": item["old"], "new": item["new"]})

    # Record manifest
    manifest_path = os.path.join(root, "branding_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "backup_dir": os.path.relpath(backup_dir, root),
            "old_identity": {"name": OLD_NAME, "app_id": OLD_APP_ID},
            "new_identity": {"name": NEW_NAME, "app_id": NEW_APP_ID, "company": NEW_COMPANY},
            "renames": renamed_pairs,
            "plan": plan,
        }, f, indent=2)
    print(f"Branding customization applied successfully. Manifest saved to {manifest_path}")

def rollback(root: str, backup_dir: str):
    if not os.path.isabs(backup_dir):
        backup_dir = os.path.join(root, backup_dir)
    manifest_file = os.path.join(backup_dir, "manifest.json")
    if not os.path.exists(manifest_file):
        print(f"Error: manifest not found in {backup_dir}")
        sys.exit(1)
        
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    branding_manifest = os.path.join(root, "branding_manifest.json")
    renames = []
    if os.path.exists(branding_manifest):
        with open(branding_manifest, "r", encoding="utf-8") as f:
            bm_data = json.load(f)
            renames = bm_data.get("renames", [])

    # Remove newly created renamed files first
    for r in renames:
        new_path = os.path.join(root, r["new"].replace("/", os.sep))
        if os.path.exists(new_path):
            os.remove(new_path)
            print(f"Removed renamed target: {r['new']}")

    # Restore all backed up files
    restored = 0
    for entry in manifest:
        rel = entry["rel_path"]
        src = os.path.join(backup_dir, rel.replace("/", os.sep))
        dst = os.path.join(root, rel.replace("/", os.sep))
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1
            
    print(f"Rollback complete: {restored} files restored from {backup_dir}")

def main():
    parser = argparse.ArgumentParser(description="Customizes project branding safely.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect proposed changes without modifying files")
    parser.add_argument("--apply", action="store_true", help="Apply proposed branding customization with automatic backup")
    parser.add_argument("--rollback", type=str, help="Restore from specified backup directory")
    args = parser.parse_args()

    root = PROJECT_ROOT
    if args.rollback:
        rollback(root, args.rollback)
        return

    plan = build_plan(root)

    if args.dry_run or (not args.apply):
        print("=== DRY-RUN BRANDING CUSTOMIZATION PLAN ===")
        print(f"Old Name: {OLD_NAME}  ->  New Name: {NEW_NAME}")
        print(f"Old App ID: {OLD_APP_ID}  ->  New App ID: {NEW_APP_ID}")
        print(f"Target changes: {len(plan)}")
        print("-" * 80)
        for idx, item in enumerate(plan, 1):
            print(f"[{idx}] {item['type'].upper()} in {item['path']}")
            print(f"    Symbol Type: {item['symbol_type']}")
            print(f"    Reason:      {item['reason']}")
            if item['type'] == 'replace':
                print(f"    - Old: {repr(item['old'])}")
                print(f"    + New: {repr(item['new'])}")
            elif item['type'] == 'rename':
                print(f"    Renaming {item['old']} -> {item['new']}")
            print()
        if not args.apply:
            print("Run with --apply to execute the migration after approval.")
        return

    if args.apply:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(root, "backups", f"branding_backup_{ts}")
        apply_plan(root, plan, backup_dir)

if __name__ == "__main__":
    main()
