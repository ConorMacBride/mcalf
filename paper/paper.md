---
title: 'MCALF: Multi-Component Atmospheric Line Fitting'
tags:
  - Python
  - astronomy
  - solar physics
  - spectrum
  - spectra
  - fitting
  - absorption
  - emission
  - voigt
authors:
  - name: Conor D. MacBride
    orcid: 0000-0002-9901-8723
    affiliation: 1
  - name: David B. Jess
    orcid: 0000-0002-9155-8039
    affiliation: "1, 2"
affiliations:
 - name: Astrophysics Research Centre, School of Mathematics and Physics, Queen's University Belfast, Belfast, BT7 1NN, UK
   index: 1
 - name: Department of Physics and Astronomy, California State University Northridge, Northridge, CA 91330, U.S.A.
   index: 2
date: 19 April 2021
bibliography: paper.bib
---

# Summary

Determining accurate velocity measurements from observations of the Sun is of vital importance to solar physicists who are studying the wave dynamics in the solar atmosphere. Weak chromospheric absorption lines, due to dynamic events in the solar atmosphere, often consist of multiple spectral components. Isolating these components allows for the velocity field of the dynamic and quiescent regimes to be studied independently. However, isolating such components is particularly challenging due to the wide variety of spectral shapes present in the same dataset. `MCALF` provides a novel method and infrastructure to determine Doppler velocities in a large dataset. Each spectrum is fitted with a model adapted to its specific spectral shape.

# Statement of need

MCALF is an open-source Python package for accurately constraining velocity information from spectral imaging observations using machine learning techniques. This software package is intended to be used by solar physicists trying to extract line-of-sight (LOS) Doppler velocity information from spectral imaging observations (Stokes $I$ measurements) of the Sun. This `toolkit' can be used to define a spectral model optimised for a particular dataset.

This package is particularly suited for extracting velocity information from spectral imaging observations where the individual spectra can contain multiple spectral components. Such multiple components are typically present when active solar phenomena occur within an isolated region of the solar disk. Spectra within such a region will often have a large emission component superimposed on top of the underlying absorption spectral profile from the quiescent solar atmosphere [@Felipe:2014]. Being able to extract velocity information from such observations would provide solar physicists with a wider range of data products that can be used for science [@Stangalini:2020]. This package implements the novel approach of automated classification of spectral profiles prior to fitting a model.

A sample model is provided for an IBIS Ca $\text{\sc{ii}}$ 8542 Å spectral imaging sunspot dataset. This dataset typically contains spectra with multiple atmospheric components and this package supports the isolation of the individual components such that velocity information can be constrained for each component. The method implemented in this IBIS model has been discussed extensively in @MacBride:2020. There are also several ongoing research projects using this model to extract velocity measurements.

Using this sample model, as well as the separate base (template) model it is built upon, a custom model can easily be built for a specific dataset. The custom model can be designed to take into account the spectral shape of each particular spectrum in the dataset. By training a neural network classifier using a sample of spectra from the dataset labelled with their spectral shapes, the spectral shape of any spectrum in the dataset can be found. The fitting algorithm can then be adjusted for each spectrum based on the particular spectral shape the neural network assigned it. The `toolkit' nature of this package also allows the possibility of utilising existing machine learning classifiers, such as the ``supervised hierarchical $k$-means" classifier introduced in @Panos:2018, which clusters solar flare spectra based on their profile shape.

This package is designed to run in parallel over large data cubes, as well as in serial. As each spectrum is processed in isolation, this package scales very well across many processor cores. Numerous functions are provided to plot the results clearly, some of which are showcased in \autoref{fig:example}. The `MCALF` API also contains many useful functions which have the potential of being integrated into other Python packages. Full documentation as well as examples on how to use `MCALF` are provided at [mcalf.macbride.me](https://mcalf.macbride.me).

![An overview of some of the plotting functions that are included in `MCALF`.\label{fig:example}](figure.pdf)

# Acknowledgements

CDM would like to thank the Northern Ireland Department for the Economy for the award of a PhD studentship. DBJ wishes to thank Invest NI and Randox Laboratories Ltd. for the award of a Research and Development Grant (059RDEN-1) that allowed the computational techniques employed to be developed. DBJ would also like to thank the UK Science and Technology Facilities Council (STFC) for the consolidated grant ST/T00021X/1. The authors wish to acknowledge scientific discussions with the Waves in the Lower Solar Atmosphere (WaLSA; [www.WaLSA.team](https://www.WaLSA.team)) team, which is supported by the Research Council of Norway (project no. 262622) and the Royal Society (award no. Hooke18b/SCTM).

# References

