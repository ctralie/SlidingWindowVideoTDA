# (Quasi)periodicity in Videos Using Sliding Windows And Topology

### by [Chris Tralie] and [Jose Perea]

## ArXiV Paper Link

https://arxiv.org/pdf/1704.08382.pdf


## Installation Instructions

To run this code, you will need to compile Ripser.  This assumes that you have checked out the code with the Ripser submodule

~~~~~ bash
git clone --recursive https://github.com/ctralie/SlidingWindowVideoTDA.git
~~~~~

Go into the ``ripser'' directory, and type the following

~~~~~ bash
make ripser
make ripser-coeff
~~~~~

## Code Structure

* VideoMix/NumberedVideos: The videos used in the Mechanical Turk Experiment
* BiphonationExperiments.py: This is the main file that loops through all of the vocal folds videos and computes our periodicity/quasiperiodicity scores, saving the results at the end
* VideoTools.py: Tools for loading videos into python (wrapping around avconv), doing PCA on videos, and simulating shake and bit error noise
* FundamentalFreq.py: Tools for computing the fundamental frequency using normalized autocorrelation of diffusion maps
* SpectralMethods.py: Contains code for computing diffusion maps
* CSMSSMTools.py: Contains code for computing fast all pairs self-similarity and affinity matrices
* TDA.py: A wrapper around the [ripser] library for computing persistence diagrams of Vietoris Rips filtrations
* TDAPlotting.py: Plotting tools for persistence diagrams 
* GeometryTools.py: Implements mean shift (currently disabled)
* AlternativePeriodicityScoring.py: Our implementations of alternative methods for computing periodicity in the literature
* SyntheticCurves.py: Code for synthesizing blur motion trajectories
* ROCExperiments.py: Code for running all of the ROC experiments for periodic and quasiperiodic videos under AWGN/blur/bit error noise
* Hodge.py: An implementation of Hodge rank aggregation
* HodgeExperiments.py: Some synthetic experiments for Hodge rank aggregation
* rankVideosTDA.py: Use TDA to come up with a global ranking of the videos
* rankVideosHumanAggregate.py: Do Hodge Rank aggregation to come up with a global ranking from pairwise rankings of videos

## Acknowledgements
The authors would like to thank Juergen Neubauer,  Dimitar Deliyski, Robert Hillman, Alessandro de Alarcon, Dariush Mehta, and Stephanie Zacharias for providing videos of vocal folds.  We also thank Matt Berger at ARFL for discussions about sliding window video efficiency.  Christopher Tralie was partially supported by an NSF Graduate Fellowship NSF under grant DGF-1106401 and an NSF big data grant DKA-1447491.  Jose Perea was partially supported by the NSF under grant DMS-1622301 and DARPA under grant HR0011-16-2-003.

The authors would also like to thank [Ulrich Bauer] for providing fast code ([ripser]) to compute Vietoris Rips filtrations.  We have mirrored his code here, and the license for his code can be found in the Ripser submodule



[Chris Tralie]: <http://www.ctralie.com>
[Jose Perea]: <https://cmse.msu.edu/directory/faculty/jose-perea/>
[Ulrich Bauer]: <http://ulrich-bauer.org>
[ripser]: <https://github.com/Ripser/ripser>
