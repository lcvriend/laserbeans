# laserbeans
*Toolbox for data exploration*

This repo aims to facilitate the data exploration process. It was developed for the Central Student Administration of [Utrecht University](www.uu.nl). Check out [this notebook](https://nbviewer.jupyter.org/github/lcvriend/laserbeans/blob/master/example.ipynb) for some examples.

The repo consists of tools for:
- Selecting data
- Dealing with dates and times
- Resampling time series data
- Creating frequency tables
- Adding percentage columns to tables
- Creating interactive charts

## Abstractions
Data exploration preferably should be quick and easy. I find that some useful and reusable table and chart types, require a significant amount of data manipulation and configuration to set up. This process, then, often entails a lot of repetitive code. The purpose of this toolbox is to abstract away most of that boilerplate code. This is done by creating convenience wrappers for:
- [Pandas](https://pandas.pydata.org/)
- [Altair](https://altair-viz.github.io)

The toolbox is under development.

---
**Note** The tools are currently designed for use in a European context. This is reflected in the formatting of dates and numbers.
