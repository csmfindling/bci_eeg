#!/usr/bin/env python

import time
import configparser # this is version 2.x specific, on version 3.x it is called "configparser" and has a different API
import redis
import sys
import os
import numpy as np
import re

class redis_serial():

    def __init__(self):

        # config file
        self.config = configparser.ConfigParser()
        self.config.read('serialRedis.ini')

        # this determines how much debugging information gets printed
        # debug = config.getint('general','debug')
        debug = 2

        try:
            self.r = redis.StrictRedis(host=self.config.get('redis','hostname'), port=self.config.getint('redis','port'), db=0)
            response = self.r.client_list()
            if debug>0:
                print("Connected to redis server")
        except redis.ConnectionError:
            print("Error: cannot connect to redis server")
            exit()

    def write(self, value, key='alpha'):
        assert(key in ['alpha', 'beta'])
        self.r.set(self.config.get('output', key), value)

