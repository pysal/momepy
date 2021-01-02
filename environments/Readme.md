# momepy container

`momepy` is a library for quantitative analysis of urban form - urban morphometrics - built on top of GeoPandas, PySAL and networkX. This container provides the minimal environment needed to run momepy. 

Documentation: http://docs.momepy.org/en/stable/

Source code: https://github.com/martinfleis/momepy

Docker repository: https://hub.docker.com/repository/docker/martinfleis/momepy

If you need the full stack of geospatial Python libraries, use [`darribas/gds_env`](https://darribas.org/gds_env/) which provides the updated platform for Geographic Data Science (including momepy).

## Install
```
docker pull martinfleis/momepy:0.4
```

## Run
To start Jupyter lab instance:

```
docker run -it -p 8888:8888 -v ${PWD}:/home/jovyan/work martinfleis/momepy:0.4
```

To start a command line interface:

```
docker run -it -p 8888:8888 -v ${PWD}:/home/jovyan/work martinfleis/momepy:0.4 start.sh
```
