import os
from pathlib import Path
import pickle
import magic_card_detector as mcg

# Reference card images path can be set via environment variable MTG_CARD_IMAGES_PATH
card_images_path = os.getenv('MTG_CARD_IMAGES_PATH', './reference_card_images/LEA')

card_detector = mcg.MagicCardDetector()
card_detector.read_and_adjust_reference_images(card_images_path)

hlist = []
for image in card_detector.reference_images:
    image.original = None
    image.clahe = None
    image.adjusted = None
    hlist.append(image)

out_dir = Path('phash_data')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'new_reference_phash.dat'
with out_path.open('wb') as f:
    pickle.dump(hlist, f)