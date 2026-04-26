import json, subprocess
base = r'c:\Users\alexy\Documents\Claude_projects\Kaggle competition\bird_clef\kaggle_notebooks'

# Push a single fold. Edit FOLDS list as needed.
# Fold 3 and 4 are already running. Push fold 5 once one slot frees up.
FOLDS = [5]

for fold in FOLDS:
    meta = {
        'id': f'alexycactus/birdclef-2026-b3-fold{fold}',
        'title': f'BirdCLEF 2026 B3 Fold{fold}',
        'code_file': f'06_b3_fold{fold}.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': 'true',
        'enable_gpu': 'true',
        'enable_internet': 'true',
        'dataset_sources': [],
        'competition_sources': ['birdclef-2026'],
        'kernel_sources': [],
        'model_sources': []
    }
    with open(fr'{base}\kernel-metadata.json', 'w', encoding='ascii') as f:
        json.dump(meta, f, indent=2)
    print(f'=== Pushing fold {fold} ===')
    r = subprocess.run(
        ['kaggle', 'kernels', 'push', '-p', base, '--accelerator', 'NvidiaTeslaT4'],
        capture_output=True, text=True
    )
    print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip())
    print()
