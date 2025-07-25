# minflux-analysis

## Description

Custom Python software package for processing and analysis of MINFLUX-data (Balzarotti _et al._, Science **355**, 606-612 (2017). DOI: [10.1126/science.aak9913](https://doi.org/10.1126/science.aak9913)).
The software package comprises the modules minflux_parameters, minflux_dataIO, minflux_psf, minflux_localization, minflux_visualization, and minflux_analysis. An example workflow is given in main.py:
* First, load MINFLUX-parameter profile. All parameters are saved to a text file for documentation.
* Load MINFLUX-data using minflux_dataIO, create a grid of tip/tilt mirror positions and time stamps.
* Create PSF-model from the minflux_psf module, using either a theoretical beam shape or fitting experimental data.
* Perform localization using the minflux_localization module. Filter count traces to remove noise, then theshold them to extract emission events, and estimate emitter positions.
* Visualize MINFLUX-counts and -localizations via the minflux_visualization module. Counts can be plotted as count traces against time or as count histogram. Localizations can be visualized as scatter plot, as 2D localization histogram, as Gaussian spots or as localization traces against time.
* Further data analysis can be performed with the minflux_analysis module. Required functionality strongly depends on experiment types and scientific questions at hand. Hence, this module currently contains limited functionality, but is meant to be extended by users as desired.

## Getting Started

### Dependencies

Python 3.12.10 with packages
* matplotlib 3.10.3
* numpy 2.2.6
* scipy 1.15.2

Tested on Windows 10 and 11.

### Installing

* Download the minflux-analysis Github repository and install a Python environment with the packages listed above. We used mainly Spyder 6.0.7, managing packages with Miniforge3 25.3.0-3.

### Executing program

* Place raw data files in the minflux_data folder. Output files are saved here, too. 
* Run main.py in a python environment of your choice. Adapt workflow to your needs. 

## Authors

Jakob Vorlaufer [@jvorlauf](http://github.com/jvorlauf)

## Version History

* This is the initial release.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE.md file for details.

## Acknowledgments

Thanks to Marek Šuplata and Julia Lyudchik for intellectual input.

