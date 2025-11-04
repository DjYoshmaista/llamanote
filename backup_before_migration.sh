#!/bin/bash
# Backup script before implementing hierarchical output structure

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup_pre_migration_${BACKUP_DATE}"

echo "========================================"
echo "Pre-Migration Backup Script"
echo "========================================"
echo ""
echo "Creating backup: ${BACKUP_DIR}"
echo ""

mkdir -p "${BACKUP_DIR}"

# Backup source code
if [ -d "src" ]; then
    echo "[1/5] Backing up source code..."
    cp -r src "${BACKUP_DIR}/src"
    echo "  ✓ Source code backed up"
else
    echo "  ⚠ No src/ directory found"
fi

# Backup checkpoints
if [ -d "checkpoints" ]; then
    echo "[2/5] Backing up checkpoints..."
    cp -r checkpoints "${BACKUP_DIR}/checkpoints"
    CKPT_COUNT=$(ls -1 checkpoints/*.ckpt 2>/dev/null | wc -l)
    echo "  ✓ ${CKPT_COUNT} checkpoints backed up"
else
    echo "  ⚠ No checkpoints/ directory found"
fi

# Backup output
if [ -d "output" ]; then
    echo "[3/5] Backing up output..."
    cp -r output "${BACKUP_DIR}/output"
    OUTPUT_COUNT=$(find output -type f 2>/dev/null | wc -l)
    echo "  ✓ ${OUTPUT_COUNT} output files backed up"
else
    echo "  ⚠ No output/ directory found"
fi

# Backup config
if [ -d "config" ]; then
    echo "[4/5] Backing up configuration..."
    cp -r config "${BACKUP_DIR}/config"
    echo "  ✓ Configuration backed up"
else
    echo "  ⚠ No config/ directory found (using defaults)"
fi

# Create manifest
echo "[5/5] Creating backup manifest..."
cat > "${BACKUP_DIR}/MANIFEST.txt" <<EOF
========================================
Pre-Migration Backup Manifest
========================================

Backup created: $(date)
Git commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Git branch: $(git branch --show-current 2>/dev/null || echo "N/A")

Files backed up:
- src/ (source code)
- checkpoints/ (checkpoint files)
- output/ (generated outputs)
- config/ (configuration)

Statistics:
-----------
Checkpoint count: $(ls -1 checkpoints/*.ckpt 2>/dev/null | wc -l)
Output files: $(find output -type f 2>/dev/null | wc -l)

Disk usage:
-----------
$(du -sh src checkpoints output config 2>/dev/null | awk '{print $2 ": " $1}')

Total backup size: $(du -sh ${BACKUP_DIR} | cut -f1)

========================================
Restoration Instructions:
========================================

To restore from this backup:

1. Full restore:
   ./restore_backup.sh ${BACKUP_DIR}

2. Partial restore (code only):
   rm -rf src && cp -r ${BACKUP_DIR}/src ./

3. Partial restore (checkpoints only):
   rm -rf checkpoints && cp -r ${BACKUP_DIR}/checkpoints ./

4. Partial restore (output only):
   rm -rf output && cp -r ${BACKUP_DIR}/output ./

========================================
End of Manifest
========================================
EOF

echo "  ✓ Manifest created"
echo ""
echo "========================================"
echo "Backup Complete!"
echo "========================================"
echo ""
echo "Backup location: ${BACKUP_DIR}"
echo "Total size: $(du -sh ${BACKUP_DIR} | cut -f1)"
echo ""
echo "Manifest: ${BACKUP_DIR}/MANIFEST.txt"
echo ""
echo "To view manifest:"
echo "  cat ${BACKUP_DIR}/MANIFEST.txt"
echo ""
echo "To restore:"
echo "  ./restore_backup.sh ${BACKUP_DIR}"
echo ""
