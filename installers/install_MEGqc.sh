#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
APP_NAME="MEGqc"
INSTALL_DIR="$HOME/MEGqc"
PORTABLE_PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.10.13+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz"
ENV_DIR="$INSTALL_DIR/env"
PYTHON_BIN="$INSTALL_DIR/bin/python3.10"
DESKTOP_DIR="$HOME/Desktop"
APPDIR="$HOME/.local/share/applications"

mkdir -p "$INSTALL_DIR" "$APPDIR"

# ──────────────────────────────────────────────────────────────
# Step 1: Download portable version
# ──────────────────────────────────────────────────────────────

USE_PORTABLE=true


if $USE_PORTABLE; then
    echo "[*] Downloading portable Python 3.10..."
    cd "$INSTALL_DIR"
    wget -q --show-progress "$PORTABLE_PYTHON_URL"
    tar -xzf "$(basename "$PORTABLE_PYTHON_URL")"
    rm "$(basename "$PORTABLE_PYTHON_URL")"
    mv python/* .
    rmdir python
    PYTHON="$PYTHON_BIN"
fi

# ──────────────────────────────────────────────────────────────
# Step 2: Create virtual environment and install MEGqc
# ──────────────────────────────────────────────────────────────
echo "[*] Creating virtual environment..."
"$PYTHON" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install git+https://github.com/ANCPLabOldenburg/MEGqc.git
deactivate

# ──────────────────────────────────────────────────────────────
# Step 3: Create launcher script
# ──────────────────────────────────────────────────────────────
RUN_SCRIPT="$INSTALL_DIR/run_MEGqc.sh"
cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
source "$ENV_DIR/bin/activate"
megqc
EOF
chmod +x "$RUN_SCRIPT"

# ──────────────────────────────────────────────────────────────
# Step 4: Create uninstaller script
# ──────────────────────────────────────────────────────────────
UNINSTALL_SCRIPT="$INSTALL_DIR/uninstall_MEGqc.sh"
cat > "$UNINSTALL_SCRIPT" <<EOF
#!/usr/bin/env bash
echo "[*] Removing $INSTALL_DIR..."
rm -rf "$INSTALL_DIR"

echo "[*] Removing launchers..."
rm -f "$DESKTOP_DIR/MEGqc.desktop"
rm -f "$DESKTOP_DIR/Uninstall_MEGqc.desktop"
rm -f "$APPDIR/MEGqc.desktop"
rm -f "$APPDIR/Uninstall_MEGqc.desktop"

echo "[*] Removing PATH override from ~/.bashrc..."
sed -i '/# >>> MEGqc Python/,/# <<< MEGqc Python/d' "$HOME/.bashrc"

echo "[*] Uninstallation complete."
EOF
chmod +x "$UNINSTALL_SCRIPT"

# ──────────────────────────────────────────────────────────────
# Step 5: Create desktop launchers
# ──────────────────────────────────────────────────────────────
if [[ -d "$DESKTOP_DIR" ]]; then
    echo "[*] Creating desktop shortcuts..."

    # Run launcher
    cat > "$APPDIR/MEGqc.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MEGqc
Comment=Launch MEGqc
Exec=$RUN_SCRIPT
Terminal=true
Icon=utilities-terminal
EOF
    cp "$APPDIR/MEGqc.desktop" "$DESKTOP_DIR/"
    chmod +x "$DESKTOP_DIR/MEGqc.desktop"

    # Uninstall launcher
    cat > "$APPDIR/Uninstall_MEGqc.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Uninstall MEGqc
Comment=Completely remove MEGqc
Exec=$UNINSTALL_SCRIPT
Terminal=true
Icon=utilities-terminal
EOF
    cp "$APPDIR/Uninstall_MEGqc.desktop" "$DESKTOP_DIR/"
    chmod +x "$DESKTOP_DIR/Uninstall_MEGqc.desktop"
else
    echo "[!] Desktop directory not found. Skipping desktop icon creation."
fi

# ──────────────────────────────────────────────────────────────
# Step 6: Set portable Python as default (user-only)
# ──────────────────────────────────────────────────────────────
if ! grep -q '# >>> MEGqc Python' "$HOME/.bashrc"; then
    echo "[*] Setting portable Python as the default Python in ~/.bashrc..."
    cat >> "$HOME/.bashrc" <<EOF

# >>> MEGqc Python
export PATH="$INSTALL_DIR/bin:\$PATH"
# <<< MEGqc Python
EOF
fi

# ──────────────────────────────────────────────────────────────
# Done!
# ──────────────────────────────────────────────────────────────
echo ""
echo "✅ MEGqc was successfully installed!"
echo "→ Run it via the desktop icon or with:  bash $RUN_SCRIPT"
echo "→ Uninstall it with:                   bash $UNINSTALL_SCRIPT"
echo "→ Restart your terminal to use Python 3.10 by default."

