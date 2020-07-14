# TOAD: Trace Ordering for Anomaly Detection

Prototype implementation of TOAD as specified in tba. This tool allows temporal collective anomaly detection by applying an adapted modification of the OPTICS clustering technique. Temporal profiles for process instances are used as objects to find dense areas, which exist in abnormal regions of the representation space. Using a reference curve, denser areas are identified and returned. This is beneficial for a later analysis, as many traces can be treated at once, as a similar deviation characteristic can imply a similar reason for the deviation and therefore allow a common treatment.

If you'd like to learn more about how it works, see References below.

Brought to you by Florian Richter (richter@dbs.ifi.lmu.de).


# Usage

This is just a prototype implementation. You can just run the whole script, if you specify the parameters in the code and load an appropriate dataset as *\*.xes*. We recommend the datasets at https://data.4tu.nl/repository/collection:event_logs_real, which have been used for the evaluation of TOAD.



# References

The algorithm used by ``TOAD`` is taken directly from the original paper by Richter, Lu, Zellner, Sontheim, and Seidl. If you would like to discuss the paper, or corresponding research questions on temporal process mining (we have implemented a few other algorithms as well) please email the authors.


