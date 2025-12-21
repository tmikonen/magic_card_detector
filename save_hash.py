import os
from pathlib import Path
import pickle
import magic_card_detector as mcg

# Reference card images path can be set via environment variable MTG_CARD_IMAGES_PATH
card_images_path = os.getenv('MTG_CARD_IMAGES_PATH', './reference_card_images/LEA')

card_detector = mcg.MagicCardDetector()
hlist = card_detector.calculate_reference_hashes(card_images_path)

out_dir = Path('phash_data')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'new_reference_phash.dat'
with out_path.open('wb') as f:
    pickle.dump(hlist, f)

# Report peak memory usage (high-water mark) where available.
try:
    import resource, platform
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = usage.ru_maxrss
    if platform.system() == 'Linux':
        peak_megabytes = peak / 1024
    else:
        peak_megabytes = peak / (1024 * 1024)
    print(f'Peak memory (high-water mark): {peak_megabytes} MB')
except Exception:
        print('Could not determine peak memory usage.')