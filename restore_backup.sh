#!/bin/bash
# Restore from backup

if [ $# -lt 1 ]; then
    echo "Usage: ./restore_backup.sh <backup_directory>"
    echo ""
    echo "Available backups:"
    ls -dt backup_pre_migration_* 2>/dev/null || echo "  No backups found"
    exit 1
fi

BACKUP_DIR="$1"

if [ ! -d "${BACKUP_DIR}" ]; then
    echo "Error: Backup directory not found: ${BACKUP_DIR}"
    exit 1
fi

echo "========================================"
echo "Backup Restoration Script"
echo "========================================"
echo ""
echo "Restoring from: ${BACKUP_DIR}"
echo ""

# Show manifest
if [ -f "${BACKUP_DIR}/MANIFEST.txt" ]; then
    echo "Backup details:"
    head -n 20 "${BACKUP_DIR}/MANIFEST.txt"
    echo ""
fi

# Confirm
read -p "This will overwrite current files. Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Restoration cancelled"
    exit 0
fi

# Create safety backup of current state
SAFETY_DIR="current_state_before_restore_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "Creating safety backup of current state: ${SAFETY_DIR}"
mkdir -p "${SAFETY_DIR}"
cp -r src "${SAFETY_DIR}/" 2>/dev/null
cp -r checkpoints "${SAFETY_DIR}/" 2>/dev/null
cp -r output "${SAFETY_DIR}/" 2>/dev/null
echo "  ✓ Current state backed up"

# Restore
echo ""
echo "Restoring files..."

if [ -d "${BACKUP_DIR}/src" ]; then
    echo "  Restoring src/..."
    rm -rf src
    cp -r "${BACKUP_DIR}/src" ./
    echo "  ✓ src/ restored"
fi

if [ -d "${BACKUP_DIR}/checkpoints" ]; then
    echo "  Restoring checkpoints/..."
    rm -rf checkpoints
    cp -r "${BACKUP_DIR}/checkpoints" ./
    echo "  ✓ checkpoints/ restored"
fi

if [ -d "${BACKUP_DIR}/output" ]; then
    echo "  Restoring output/..."
    rm -rf output
    cp -r "${BACKUP_DIR}/output" ./
    echo "  ✓ output/ restored"
fi

if [ -d "${BACKUP_DIR}/config" ]; then
    echo "  Restoring config/..."
    rm -rf config
    cp -r "${BACKUP_DIR}/config" ./
    echo "  ✓ config/ restored"
fi

echo ""
echo "========================================"
echo "Restoration Complete!"
echo "========================================"
echo ""
echo "Current state saved to: ${SAFETY_DIR}"
echo ""
echo "To verify restoration:"
echo "  python -c 'from src.io.checkpoints import CheckpointManager; print(\"✓ Import successful\")'"
echo ""
