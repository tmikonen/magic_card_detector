# Magic Card Detector

This project is a Python implementation for detecting and recognizing Magic: the Gathering cards in images. It includes card segmentation and recognition algorithms, and now features a Flask web application for easy use via a browser.

## Description

This project provides a solution for automatically detecting and recognizing Magic: the Gathering cards from images. It leverages computer vision techniques for card segmentation and recognition. You can use it via command line or through a web interface. More details about the algorithms and examples can be found in the blog post: [https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/](https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/)

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

### Command Line Interface (CLI)

To use the magic card detector from the command line, run the `magic_card_detector.py` script with the path to the directory containing images you want to analyze, and specify an output directory.

```bash
python magic_card_detector.py <image_path> <output_path>
```

For example, to analyze the images in the `example` directory and save results to the `results` directory:

```bash
python magic_card_detector.py example results
```

The script will output the recognition results and save images with bounding boxes and labels in the specified output directory.

### Web Application

To use the web application, you need to run the Flask app first.

1.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    This will start the Flask development server. By default, it will be accessible at `http://127.0.0.1:5001` or `http://localhost:5001`.

2.  **Open in your browser:**
    Navigate to the address provided in the terminal output (usually `http://127.0.0.1:5001`) in your web browser.

3.  **Use the web interface:**
    You will see a simple web page where you can upload an image of Magic: the Gathering cards. After uploading and submitting the image, the detector will process it and display the recognition results on a new page.


## Project Structure

```
.
├── .gitignore
├── .pylintrc
├── alpha_reference_phash.dat
├── app.py
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
├── save_hash.py
└── templates/
    ├── index.html
    └── results.html
```

* `magic_card_detector.py`: Main script for card detection and recognition (CLI).
* `save_hash.py`: Script for precalculating and saving image hashes.
* `app.py`: Flask application for the web interface.
* `templates/`: Contains HTML templates for the web application.
* `example/`: Contains example images for testing.
* `results/`:  Expected results from running the script on the example images.
* `alpha_reference_phash.dat`: Hash data for Limited Edition Alpha cards.
* `requirements.txt`: Lists the Python dependencies.
* `README.md`: This file, providing project information.
* `LICENSE`: License file.

## License

This project is licensed under the [MIT License](LICENSE).
