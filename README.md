# G2Net Gravitational Wave Detection
https://github.com/rohan-paul/Gravitational-Wave-Detection_Kaggle_Competition/blob/main/Kaggle_NBs/1_TimeSeries_GWPy_Data_Preprocessing.ipynb

This is the final project done for Big Data and Data Science UCM master

In this competition you are provided with a training set of time sereis data containing simulated gravitational wave measurements from a network of 3 gravitational wave interferometers (LIGO Hanford, LIGO Livingston and Virgo). Each time series contains either detector noise or detector noise plus a simulated gravitational wave signal. The objective is to identify when a signal is present in the data (target= 1).

So we need to use the trainin data along with the target value to build our model and make predictions on the test IDs in form of probability that the target exists for that ID.

So basically data-science helping here by building models to filter out this noises from data-strams (which includes both noise frequencies and Gravitational Weaves frequencies) so we can single out frequencies for Gravitational-Waves. 

## Basic Description of the Data Provided

We are provided with a train and test set of tiem series data containing simulated gravitational wave measuraments from a network of 3 gravitational wave interferometers:

- LIGO Handford
- LIGo Livinston
- Virgo

Each time series contains either detector noise or detector nnoise plus a simulated gravitational wave signal.
The task is to identify when a signal is present in the data (target=1).,

Each .npy data file contains 3 times series (1 coming for each detector) and each spans 2 sec and is sampled at 2,048 Hz.

And we have a total of 5,600,000 files, each file of dimension of 3*4096, this turns out to be a huge time series.

## What are Gravitational Waves? https://www.ligo.caltech.edu/page/what-are-gw

Gravitational waves are "riples" in space-time caused by some of the most violent and energetic processes in the Universe. Albert Einstein predicted the existence of gravitational waves in 1916 in his general theory of relativity. Einstein's's mathematics showed that massive accelerating objects (such as neutron stars or black holes orbiting each other) would disrupt space-time in such a way that "waves" of undulating space-time woulkd propagate in all directions away from the source. These cosmic ripples would travel at the speed of light, carrying with them information about their origins, as well as clues to the nature of gravity itself.

The strongest gravitational waves are produced by cataclysmic events such as colliding black holes, supernovae, and colliding neutron stars. Other waves are predicted to be caused by the rotation of neutron stars that are not perfect shperes, and possibly even the remnants of gravitational radiation created by the Big Bang

On September 14, 2015 for the very first time, LIGO physically sensed the undulations in spacetime caused by gravitational waves generated by two colliding black holes 1.3 billion light-years away. LIGO's discovery will go down in history as one of humanity's greatest scientific achievements

While the processes that general gravitaitonl wavces can be extremely violent and edestructive, by the time the waves reach Earth they are thousands of billions of times smaller. In fact, by the time gravitational waves from LIGO's first detection reached us, the amount of space-time wobbling they generated was a 1000 times smaller than athe nucleus of an atom. 

Gravitational wave data analyisis employs many of the standard tools of time-series analysis. With some exceptions, the majority of gravitational wave data analysis is performed in the frequency domain.

Data from gravitational wave detectors are recorded as time series that include contributions from myriad noise sources in addition to any gravitational wave signals. When regularly sampled data are aailable, such as for ground based and future space based interferometers, analyses are typically performed in the frequency domain, where stationary (time invariant) noise processes can be modeled very efficiently.

As shown bt Einstein himself, the General Relativbity Theory of Gravitation allows fro wave-like solutions which are generated by accelerated mass motion ( or, more generally,m by energy momentum tensor as should be for a relativistic theory).


## Time Domain vs Frequency Domain Analysis?

A Time domain analysisis is an analysis of physical signals, mathematicals functions, or time series of economic or environmental data, in reference to time. Also, in the time domain, the signal or function0s value is understood for all real numbers at various separate instances in the case of discrete-time or the case of continuous-time. Furthermore, an oscilloscope is a tool commonly used to see real-world signals in the time domain.

Moreover, a time-domain graph can show how a signal changes with time

In Frequency domain your model/system is analyzed according to it's response for different frequencies. How much of the signal lie in different frequency range. Theoretically signals are composed of many sinusoidal signals with different frequencies (Fourier series), like triangle signal, its acutally composed of infinite sinusoidal singals (fundamental and odd harmonic frequencies).

We can move from time domain to frequency domain with the help of Fourir transform.

## The problem of glitches in LIGO data and how Data Science can help

Modern interferometric gravitational-wave (GW) detectors are highly complex and sensitive instruments. Each detector is sensitive not only to gravitational radiation, but also to noise from sources including the physical environment, seismic activity, and complications in the detector itself. The output data of these detectors is therefore also highly complex. In addition to the desired signal, the GW data stream contains sharp lines in its noise spectrum and non-Gaussian transients, or "glitches" that are not astrophysical in origin. Instrumental artifgacts in the GW can be mistaken for short-duration, unmodeled GW events, and noisy data can also decrease the confidence in compact binay detections, sometimes by orders of magnitude.

References and Further Reading on Gravitational Waves
https://www.nature.com/articles/d41586-018-04157-6

https://dcc.ligo.org/public/0122/P1500260/015/errata_authors_Martynov_PRD_AF.pdf - Has good Noise sensitivity data

## What are .npy files?
It is a standard binary file format for persisting a single arbitrary NUmPOy array on disk. The format stores all the shape and data type information necessaru to reconstruct the array correctly even on another machine with a different architecture. The format is designed to be as simple as possible while achieving its limited goals. THe impolementation is intended to be pure Python and distributed as part of the main numpy package.


https://github.com/SiddharthPatel45/gravitational-wave-detection/blob/main/code/gw-detection-analysis.ipynb

Gravitational Waves have been discussed since the beginning of the 20th century, and scientifically researched since the Einsten's General Theory of Relativity. They are caused by massive celestal bodies, like a Neutron Star or Black Holes, when they accelerate they cause gravitational waves, in the form of waves, propagating through the curvature of space-time at the speed of light. These distrubances can be felt on the other side of the observable universe, but are extremely weak as they lose energy as gravitational radiation. It can be imagined similar to throwing a pebble in the pond, the site where the pebble hits water is the source of the distrubance and the outgoing ripples, are the gravitational waves, that get weaker as they move away from the source.

The gravitational waves opens new "windows" to observe and study the events of the universe, which were not possible with the electromagnetic radiation that we usually rely on, using the radio and optical telescopes. These waves travel through the universe without interacting with matter and thus virtually undisturbed. Detecting them can unpack a lot of information about their origins and how our universe works.

They can be detected directly or indirectly. In the 1970s, scientists determined that, for that to happen, the system must be radiating energy in form of gravitational waves. Pulsars are a highly magnetized rotating compact star that emits beams of electromagnetic raditation out of its magnetic poles.

It was only in late 2015, that the LIGO team announced the first direct detection of gravitational wave, caused by merger of two black holes, using ultra-sensitive ground-based laser insturments. This lead to confirmation of Einstein's predictions almost a century ago.

The highly sensitive and precise laser interferometers, measuere tiny the ripples caused by the gravitational waves by superimposing two laser beams orthogonally, and recording the phase difference as strain. This happens because the gravitational waves stretch and contract the space when they move through it. These measurements are extremely tiny and are very susceptible to surrouding distrubances like vibration from nearby equipments, seismic atictivyt, etc. That's where Machine Learning comes in, as the signals are buried in detector noise.

G2Net is a network of Gravitational Waves, Geophysics and Machine Learning. With the increasing interest in machine learning and its applications like data handling, data analysis & visualizations, prediction and classification capabilities, among many more, can we leverage these for noise removal, data conditioning tools, and signal characterization. Deep Learning, especially, has proved really effective at asolving such problems where complex computations can be replaced by a well-trained model and focused for preditcting the future.

## Problem Statement

Build a Machine Learning pipeline to read, preprocess, train models and predict the gravitational wave signals. Since it is really difficult to tell the samples with and without GW signals apart, use ROC AUC metric to build the best classifier.

This project is based on a recent kaggle competition. Finding a project in physics has been something I've been looking into since I got into data science, and this challenge is the perfect candidate for fulfilling a personal interest while learning a lot about the astrophysical phenomenon and signal processing, in particular. For all the reasons the discovery of Gravitational Waves is important, it is also not a easy feat to manage this project, e.g., the entire dataset is 72GB.

The main objective here is to build a modelling pipeline that can handle such a large dataset and flexible enough to improve on in the future.

Since this is a kaggle competition, and we need GPU access to reduce the computation times, we use the kaggle's notebook environment to train our depp learning model. For initial analysis, we create a separate notebook which can be run on any compatible machine. There are 786,000 mesuraments in total, out of which 560,000 are training samples and remainin belong to the test, on which we need to make our final predictions to submit on kaggle. Each of this is an ID which has it's namesake data file, each of which contains three time series, one of every observatory.

The quantity in this time series is strain, which is of the order of 10^-20, recorded for 2 sec periods sampled at 2048 Hz. Some of these "waves" contain the signal and all of them contain noise, which is not always visible to the eye. To tackle this, we follow signal processing methodology to preprocess signals, converting the time domain data to frequency domain, converting to Constant Q-Transform images and using these as input to our model trainin step. We also follow some gravitational wave detection tutorials.

There are mainly two ways in which we can preprocess this type of data to train our models:

1. Using the time series data, and performing some cleaning steps to enhance the signal, remove noise, as desbrivedadada
2. Getting the COnstant Q- Transformed spectogram image, which is a frequency-domain fourier transformed data, while trating the sample being analyzed as a wave.

From both of above steps, this project had more success with the second approach. 