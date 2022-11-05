"""
MIT License

Copyright (c) 2021 Timo Ikonen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------
Any additionally changes made, not included in the original software do not have the rights
extended under any license
"""
import pickle
import src.card_image_detector.magic_card_detector as mcg


card_detector = mcg.MagicCardDetector()
card_detector.read_and_adjust_reference_images('../../MTG/Card_Images/LEA/')

hlist = []
for image in card_detector.reference_images:
    image.original = None
    image.clahe = None
    image.adjusted = None
    hlist.append(image)

with open('reference_hashes/alpha_reference_phash.dat', 'wb') as f:
    pickle.dump(hlist, f)
