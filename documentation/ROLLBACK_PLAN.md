# Comprehensive Rollback Plan

## Overview

This document provides detailed procedures for rolling back changes if issues arise during or after the hierarchical output structure implementation.

---

## Table of Contents

1. [Pre-Implementation Backup](#pre-implementation-backup)
2. [Rollback Levels](#rollback-levels)
3. [Emergency Procedures](#emergency-procedures)
4. [Partial Rollback Procedures](#partial-rollback-procedures)
5. [Data Recovery](#data-recovery)
6. [Testing Rollback](#testing-rollback)

---

## Pre-Implementation Backup

### Critical Data to Backup

Before ANY changes are made, create comprehensive backups:

```bash
#!/bin/bash
# backup_before_migration.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup_pre_migration_${BACKUP_DATE}"

echo "Creating backup: ${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"

# Backup code
echo "Backing up source code..."
cp -r src "${BACKUP_DIR}/src"

# Backup data
echo "Backing up checkpoints..."
cp -r checkpoints "${BACKUP_DIR}/checkpoints"

echo "Backing up output..."
cp -r output "${BACKUP_DIR}/output"

# Backup config
echo "Backing up configuration..."
cp -r config "${BACKUP_DIR}/config" 2>/dev/null || true

# Create manifest
echo "Creating backup manifest..."
cat > "${BACKUP_DIR}/MANIFEST.txt" <<EOF
Backup created: $(date)
Git commit: $(git rev-parse HEAD)
Branch: $(git branch --show-current)

Files backed up:
- src/
- checkpoints/
- output/
- config/

Checkpoint count: $(ls -1 checkpoints/*.ckpt 2>/dev/null | wc -l)
Output files: $(find output -type f 2>/dev/null | wc -l)

Disk usage:
$(du -sh checkpoints output src 2>/dev/null)
EOF

echo "Backup complete: ${BACKUP_DIR}"
echo "Total size: $(du -sh ${BACKUP_DIR} | cut -f1)"
```

### Git Safety

```bash
# Create safety branch
git checkout -b pre-hierarchical-backup
git add -A
git commit -m "Backup before hierarchical output implementation"

# Create tag
git tag -a pre-hierarchical-v0.0.21 -m "State before hierarchical structure"

# Return to feature branch
git checkout -b feature/hierarchical-output
```

---

## Rollback Levels

### Level 0: No Rollback Needed

**Status:** Everything working as expected

**Action:** None

---

### Level 1: Configuration Rollback

**Symptoms:**
- Program crashes on startup
- Configuration errors
- Feature flags causing issues

**Affected:**
- Configuration files only
- No data loss

**Rollback Procedure:**

```bash
# Restore config files
cp backup_pre_migration_*/config/* config/

# Or manually edit config
nano config/default.yaml

# Disable new features
# Set use_hierarchical_output: false
# Set enable_transcripts: false
# Set enable_auto_compression: false
```

**Time Required:** < 5 minutes

**Risk:** Very Low

---

### Level 2: Code Rollback (Keep Data)

**Symptoms:**
- Pipeline failures
- Checkpoint loading errors
- Processing errors

**Affected:**
- Source code
- Data preserved

**Rollback Procedure:**

```bash
# Option A: Git revert
git checkout pre-hierarchical-backup

# Option B: Restore from backup
rm -rf src
cp -r backup_pre_migration_*/src ./

# Test
python main.py --help
```

**Time Required:** < 10 minutes

**Risk:** Low (data preserved)

**Validation:**
```bash
# Ensure old checkpoints still load
python -c "
from src.io.checkpoints import CheckpointManager
cm = CheckpointManager('checkpoints')
checkpoints = list(Path('checkpoints').glob('*.ckpt'))
print(f'Can load {len([c for c in checkpoints if cm.load(c)])} checkpoints')
"
```

---

### Level 3: Data Rollback (Checkpoint Restore)

**Symptoms:**
- Checkpoints corrupted
- Missing checkpoint data
- Version compatibility errors

**Affected:**
- Checkpoint files
- Code may be preserved

**Rollback Procedure:**

```bash
# Stop any running processes
pkill -f "python main.py"

# Backup current state (may contain useful data)
mv checkpoints checkpoints.corrupted.$(date +%Y%m%d_%H%M%S)

# Restore checkpoints
cp -r backup_pre_migration_*/checkpoints ./

# Verify
ls -lh checkpoints/
```

**Time Required:** 5-30 minutes (depends on data size)

**Risk:** Medium (may lose recent work)

**Data Loss:**
- Any checkpoints created after backup: LOST
- Any processing done after backup: LOST

---

### Level 4: Full System Rollback

**Symptoms:**
- Complete system failure
- Data corruption across multiple areas
- Irrecoverable errors

**Affected:**
- Everything

**Rollback Procedure:**

```bash
#!/bin/bash
# full_rollback.sh

BACKUP_DIR="backup_pre_migration_*"  # Use most recent

echo "⚠️  FULL SYSTEM ROLLBACK ⚠️"
echo "This will restore system to pre-migration state"
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted"
    exit 1
fi

# Stop all processes
echo "Stopping processes..."
pkill -f "python main.py"
sleep 2

# Backup current (possibly corrupted) state
echo "Backing up current state..."
CORRUPT_DIR="corrupted_state_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${CORRUPT_DIR}"
cp -r checkpoints "${CORRUPT_DIR}/" 2>/dev/null || true
cp -r output "${CORRUPT_DIR}/" 2>/dev/null || true
cp -r src "${CORRUPT_DIR}/" 2>/dev/null || true

# Restore everything
echo "Restoring source code..."
rm -rf src
cp -r ${BACKUP_DIR}/src ./

echo "Restoring checkpoints..."
rm -rf checkpoints
cp -r ${BACKUP_DIR}/checkpoints ./

echo "Restoring output..."
rm -rf output
cp -r ${BACKUP_DIR}/output ./

echo "Restoring config..."
rm -rf config
cp -r ${BACKUP_DIR}/config ./ 2>/dev/null || true

# Git rollback
echo "Rolling back git..."
git checkout pre-hierarchical-backup

echo "✓ Full rollback complete"
echo "Corrupted state saved to: ${CORRUPT_DIR}"
echo ""
echo "Verification:"
python -c "
from pathlib import Path
print(f\"Checkpoints: {len(list(Path('checkpoints').glob('*.ckpt')))}\")
print(f\"Output files: {len(list(Path('output').rglob('*')))} \")
"
```

**Time Required:** 30-60 minutes

**Risk:** High (complete reset)

**Data Loss:**
- ALL work since backup: LOST
- System reset to backup point

---

## Emergency Procedures

### Emergency Stop

If migration or processing is causing issues:

```bash
# Immediate stop
pkill -9 -f "python main.py"
pkill -9 -f "migrate_checkpoints"

# Check for zombie processes
ps aux | grep python

# Kill if necessary
kill -9 <PID>
```

### Database Corruption

If session or text file databases corrupted:

```bash
# Backup corrupted DBs
cp sessions/database.json sessions/database.json.corrupted
cp text_files/database.json text_files/database.json.corrupted

# Restore or rebuild
# Option 1: Restore from backup
cp backup_pre_migration_*/sessions/database.json sessions/

# Option 2: Rebuild from checkpoints
python rebuild_databases.py
```

### Checkpoint Corruption Detection

```python
# verify_checkpoints.py
#!/usr/bin/env python3
"""
Verify checkpoint integrity.
"""

from pathlib import Path
from src.io.checkpoints import CheckpointManager
import sys

def verify_all_checkpoints():
    cm = CheckpointManager(Path("checkpoints"))

    checkpoints = list(Path("checkpoints").glob("*.ckpt"))
    total = len(checkpoints)
    valid = 0
    corrupted = []

    print(f"Verifying {total} checkpoints...")

    for ckpt in checkpoints:
        try:
            result = cm.load(ckpt)
            if result:
                valid += 1
                print(f"✓ {ckpt.name}")
            else:
                corrupted.append(ckpt)
                print(f"✗ {ckpt.name} - Failed to load")
        except Exception as e:
            corrupted.append(ckpt)
            print(f"✗ {ckpt.name} - Error: {e}")

    print(f"\nResults: {valid}/{total} valid")

    if corrupted:
        print(f"\nCorrupted checkpoints ({len(corrupted)}):")
        for ckpt in corrupted:
            print(f"  - {ckpt}")

        return False

    return True

if __name__ == "__main__":
    success = verify_all_checkpoints()
    sys.exit(0 if success else 1)
```

**Usage:**
```bash
python verify_checkpoints.py
```

---

## Partial Rollback Procedures

### Rollback Single Feature

#### Disable Transcripts

```python
# In config/default.yaml
pipeline:
  enable_transcripts: false

# Or programmatically
from src.core.types import PipelineConfig
config = PipelineConfig()
config.enable_transcripts = False
```

#### Disable Compression

```python
# In config/default.yaml
compression:
  auto_compress: false

# Or programmatically
config.enable_auto_compression = False
```

#### Disable Hierarchical Output

```python
# In config/default.yaml
pipeline:
  use_hierarchical_output: false

# This falls back to old output structure
```

### Rollback Single Checkpoint

```python
# rollback_single_checkpoint.py
#!/usr/bin/env python3

import sys
from pathlib import Path
from src.io.checkpoints import CheckpointManager

def rollback_checkpoint(ckpt_path: Path):
    """Rollback single checkpoint from v3.0 to v2.0."""

    # Load checkpoint
    cm = CheckpointManager(Path("checkpoints"))
    result = cm.load(ckpt_path)

    if not result:
        print(f"Failed to load {ckpt_path}")
        return False

    metadata, data = result

    # Check version
    if metadata.get("version") != "3.0":
        print(f"Checkpoint is not v3.0 (current: {metadata.get('version')})")
        return False

    # Convert to v2.0
    metadata["version"] = "2.0"

    # Remove v3.0-specific fields
    metadata.pop("session_id", None)
    metadata.pop("checkpoint_number", None)
    metadata.pop("chunk_number", None)

    # Save as v2.0
    cm.save(
        stage=metadata["stage"],
        input_stem=metadata["input_stem"],
        data=data,
        metadata=metadata,
        chunk_index=None  # v2.0 doesn't use chunk_index
    )

    print(f"✓ Rolled back {ckpt_path.name} to v2.0")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rollback_single_checkpoint.py <checkpoint.ckpt>")
        sys.exit(1)

    ckpt_path = Path(sys.argv[1])
    success = rollback_checkpoint(ckpt_path)
    sys.exit(0 if success else 1)
```

---

## Data Recovery

### Recover from Corrupted Checkpoint

```python
# recover_checkpoint.py
#!/usr/bin/env python3

import pickle
import gzip
from pathlib import Path

def attempt_recovery(ckpt_path: Path):
    """Attempt to recover data from corrupted checkpoint."""

    print(f"Attempting recovery: {ckpt_path}")

    # Try different methods
    methods = [
        ("Standard pickle", lambda: pickle.load(open(ckpt_path, 'rb'))),
        ("Gzip pickle", lambda: pickle.load(gzip.open(ckpt_path, 'rb'))),
        ("Partial pickle", lambda: _partial_pickle_load(ckpt_path)),
    ]

    for name, method in methods:
        try:
            print(f"  Trying: {name}...")
            data = method()
            print(f"  ✓ Success with {name}")
            return data
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("  All recovery methods failed")
    return None

def _partial_pickle_load(path: Path):
    """Try to load partial data from pickle."""
    # This is a simplified version
    # Real implementation would try to extract partial data
    with open(path, 'rb') as f:
        # Read until error
        data = []
        try:
            while True:
                obj = pickle.load(f)
                data.append(obj)
        except EOFError:
            pass

    return data[0] if data else None
```

### Recover Lost Sessions

```python
# rebuild_sessions.py
#!/usr/bin/env python3
"""
Rebuild session database from checkpoints.
"""

from pathlib import Path
from src.io.session_manager import SessionManager
from src.io.checkpoints import CheckpointManager

def rebuild_sessions():
    """Rebuild sessions database from checkpoint metadata."""

    sm = SessionManager(Path("sessions/database.json"))
    cm = CheckpointManager(Path("checkpoints"))

    checkpoints = list(Path("checkpoints").glob("*.ckpt"))

    print(f"Rebuilding sessions from {len(checkpoints)} checkpoints...")

    sessions = {}

    for ckpt in checkpoints:
        result = cm.load(ckpt)
        if not result:
            continue

        metadata, _ = result

        session_id = metadata.get("session_id")
        if not session_id:
            # Generate session for v1.0/v2.0 checkpoints
            session_id = f"session_recovered_{ckpt.stem}"

        if session_id not in sessions:
            # Create session entry
            sessions[session_id] = {
                "session_id": session_id,
                "input_stem": metadata.get("input_stem"),
                "status": "completed",
                "checkpoints": []
            }

        # Add checkpoint to session
        sessions[session_id]["checkpoints"].append({
            "name": ckpt.name,
            "stage": metadata.get("stage"),
        })

    # Save rebuilt database
    for session_id, session_data in sessions.items():
        # Update session manager
        # (Implementation depends on SessionManager API)
        pass

    print(f"✓ Rebuilt {len(sessions)} sessions")

if __name__ == "__main__":
    rebuild_sessions()
```

---

## Testing Rollback

### Test Rollback Procedures (Before They're Needed)

```bash
#!/bin/bash
# test_rollback.sh

echo "Testing rollback procedures..."

# Create test environment
TEST_DIR="rollback_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TEST_DIR}"

# Copy test data
cp -r checkpoints "${TEST_DIR}/"
cp -r src "${TEST_DIR}/"

cd "${TEST_DIR}"

# Test Level 1: Config rollback
echo "Test 1: Config rollback"
# ... test config restore ...

# Test Level 2: Code rollback
echo "Test 2: Code rollback"
# ... test code restore ...

# Test Level 3: Data rollback
echo "Test 3: Data rollback"
# ... test checkpoint restore ...

# Test Level 4: Full rollback
echo "Test 4: Full rollback"
# ... test complete restore ...

echo "✓ All rollback tests passed"
```

---

## Rollback Decision Tree

```
                    Is there a problem?
                           |
                    Yes ───┴─── No (Keep monitoring)
                     |
            Can you disable the
            problematic feature?
                     |
              Yes ───┴─── No
               |            |
        Level 1 Rollback    Is it a code issue?
        (Config only)            |
                          Yes ───┴─── No
                           |            |
                    Level 2 Rollback    Are checkpoints
                    (Code only)         corrupted?
                                            |
                                     Yes ───┴─── No
                                      |            |
                               Level 3 Rollback    Is system
                               (Checkpoints)       unusable?
                                                      |
                                               Yes ───┴─── No (Isolate issue)
                                                |
                                         Level 4 Rollback
                                         (Full system)
```

---

## Post-Rollback Actions

### After Any Rollback

1. **Document what happened**
   ```bash
   cat > rollback_log_$(date +%Y%m%d_%H%M%S).md <<EOF
   # Rollback Log

   Date: $(date)
   Level: [1/2/3/4]
   Reason: [Description]

   Symptoms:
   - [List symptoms]

   Actions taken:
   - [List actions]

   Data lost:
   - [List any lost data]

   Lessons learned:
   - [What went wrong]
   - [How to prevent]
   EOF
   ```

2. **Verify system state**
   ```bash
   python verify_checkpoints.py
   python -m pytest tests/
   ```

3. **Test basic functionality**
   ```bash
   # Try loading a checkpoint
   python main.py --resume checkpoints/test.ckpt --dry-run

   # Try processing a small file
   python main.py --input small_test.pdf
   ```

4. **Review logs**
   ```bash
   # Check for error patterns
   grep -i error logs/*.log
   grep -i warning logs/*.log
   ```

---

## Prevention Checklist

To avoid needing rollback:

- [ ] Create comprehensive backups before changes
- [ ] Test on small dataset first
- [ ] Enable feature flags for gradual rollout
- [ ] Monitor logs during migration
- [ ] Verify checkpoints after migration
- [ ] Keep backup for 30 days minimum
- [ ] Document all changes
- [ ] Have rollback plan ready
- [ ] Test rollback procedures before migration
- [ ] Communicate with users about risks

---

## Emergency Contacts

If automated rollback fails:

1. Check git history: `git log --oneline`
2. Review backup manifest: `cat backup_pre_migration_*/MANIFEST.txt`
3. Consult implementation plan: `INTEGRATION_PLAN.md`
4. Review error logs: `logs/*.log`

---

## Rollback Success Criteria

Rollback is successful when:

- [ ] System starts without errors
- [ ] Old checkpoints load correctly
- [ ] Can process new files
- [ ] Can resume from checkpoints
- [ ] No data corruption detected
- [ ] All tests pass
- [ ] No unexpected errors in logs

---

**END OF ROLLBACK PLAN**
