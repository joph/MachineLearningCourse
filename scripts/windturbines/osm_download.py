import overpy

import fiona
from fiona.crs import from_epsg

from shapely.geometry import Point, mapping
import re

import argparse

import math
import time

from pathlib import Path

import os

import pandas as pd


def read_params():
    
    config_file = Path(__file__).parent.parent.parent / "config"
    
    
    files = [x for x in os.listdir(config_file) if (x.endswith(".csv") and x.startswith("params"))]
    p = {}
    
    countries = []
    
    for i in files:
        country = i[6:-4]
        countries.append(country)
        
        p[country] = pd.read_csv(config_file / i)
    
    return((countries,p))
        
COUNTRIES, PARAMS = read_params()

def get_param(country,name):
    val=PARAMS[country].at[0,name]
    return(val)


def distance(lon1, lat1, lon2, lat2):  # returns the distance via the haversine formula (km)
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) * \
                 math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return(d)

def getfeatures(ISOCountry, tagkey, tagval):
    
    print("Starting download...")
    
    api = overpy.Overpass()

    st = time.time()

    res = api.query("""
        area["ISO3166-1"="""+ISOCountry+"""][admin_level=2];
        node(area)["""+tagkey+"""~"""+tagval+"""];
        out center;
    """)

    nodes = res.nodes

    et = time.time()

    print('waiting for {:.0f} seconds'.format(et - st))

    time.sleep(et - st)

    st = time.time()

    res = api.query("""
        area["ISO3166-1"="""+ISOCountry+"""][admin_level=2];
        way(area)["""+tagkey+"""~"""+tagval+"""];
        out center;
    """)

    ways = res.ways

    et = time.time()

    print('waiting for {:.0f} seconds'.format(et - st))

    time.sleep(et - st)

    res = api.query("""
        area["ISO3166-1"="""+ISOCountry+"""][admin_level=2];
        relation(area)["""+tagkey+"""~"""+tagval+"""];
        out center;
    """)

    rels = res.relations

    print('\tnodes: {}, ways: {}, relations: {}'.format(len(nodes), len(ways), len(rels)))

    return nodes, ways, rels

def writeshape(nodes, name):

    schema = { 'geometry': 'Point', 'properties': {'method': 'str', 'source': 'str', 'elmw': 'float'} }
    shapeout = name

    with fiona.open(shapeout, 'w', crs=from_epsg(4326), driver='ESRI Shapefile', schema=schema) as output:
        for node in nodes:
            gensource = node.tags.get("generator:source", "unknown")
            genmethod = node.tags.get("generator:method", "unknown")
            genmodel = node.tags.get("generator:model", "unknown")

            genoutel = node.tags.get("generator:output:electricity", "unknown")
            genoutmw = -9999
            if re.search("[+-]?([0-9]*[.])?[0-9]+", genoutel):
                genoutmw = float(re.search("[+-]?([0-9]*[.])?[0-9]+", genoutel)[0])
                if "kw" in genoutel.lower():
                    genoutmw = genoutmw / 1000
            try:
                point = Point(node.lon, node.lat)
            except AttributeError:
                point = Point(node.center_lon, node.center_lat)
            prop = {
                'method': genmethod.lower(),
                'source': gensource.lower(),
                'elmw': genoutmw}
            output.write({'geometry': mapping(point), 'properties': prop})

def checkdub():
    count = 0
    for node in nodes:
        for node2 in nodes:
            dist = distance(node.lon, node.lat, node2.lon, node2.lat)
            if (node.tags.get("generator:source", "unkown") == 'wind') & (0 < dist < 0.1) & (
                    node.tags.get("generator:source", "unkown") == node2.tags.get("generator:source", "unkown")):
                count += 1
                print(
                    "NEAR: ",
                    count,
                    dist,
                    node.lon,
                    node.lat,
                    node2.lon,
                    node2.lat)
                if node2.tags.get("generator:output:electricity",
                                  "unkown") == "unkown":
                    nodes.remove(node2)
                    print("remove node2")
                else:
                    break
        else:
            continue
        nodes.remove(node)
        print("remove node1")
        break


