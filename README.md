# Magic Card Detector

This project is a Python implementation for detecting and recognizing Magic: the Gathering cards in images. It includes card segmentation and recognition algorithms.

## Description

This project provides a solution for automatically detecting and recognizing Magic: the Gathering cards from images. It leverages computer vision techniques for card segmentation and recognition.  More details about the algorithms and examples can be found in the blog post: [https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/](https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tmikonen/magic_card_detector.git
   cd magic_card_detector
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

As of April 2025, the dependencies and their versions have been confirmed for Python 3.13.2.

## Usage

To use the magic card detector, run the `magic_card_detector.py` script with the path to the directory containing the image(s) you want to analyze.

```bash
python magic_card_detector.py <image_path> <output_path>
```

For example, to analyze the images in the `example` directory:

```bash
python magic_card_detector.py example results
```

The script will output the recognition results and save an image with bounding boxes and labels in the `results/` directory.

## Project Structure

```
.
├── .gitignore
├── .pylintrc
├── alpha_reference_phash.dat
├── doc
│   ├── data_flow_diagram.md
│   ├── data_flow_diagram.png
├── example/
│   ├── alpha_deck.jpg
│   ├── black.jpg
│   ├── counterspell_bgs.jpg
│   ├── dragon_whelp.jpg
│   ├── geyser_twister_fireball.jpg
│   ├── instill_energy.jpg
│   ├── lands_and_fatties.jpg
│   └── ruby.jpg
├── LICENSE
├── magic_card_detector.py
├── README.md
├── requirements.txt
├── results/
│   ├── MTG_card_recognition_results_alpha_deck.jpg
│   ├── MTG_card_recognition_results_black.jpg
│   ├── MTG_card_recognition_results_counterspell_bgs.jpg
│   ├── MTG_card_recognition_results_dragon_whelp.jpg
│   ├── MTG_card_recognition_results_geyser_twister_fireball.jpg
│   ├── MTG_card_recognition_results_instill_energy.jpg
│   ├── MTG_card_recognition_results_lands_and_fatties.jpg
│   └── MTG_card_recognition_results_ruby.jpg
└── save_hash.py
```

* `magic_card_detector.py`: Main script for card detection and recognition.
* `example/`: Contains example images for testing.
* `results/`:  Expected results from running the script on the example images.
* `alpha_reference_phash.dat`: Hash data for Limited Edition Alpha cards.
* `requirements.txt`: Lists the Python dependencies.
* `README.md`: This file, providing project information.
* `LICENSE`: License file.

## License

This project is licensed under the [MIT License](LICENSE).
