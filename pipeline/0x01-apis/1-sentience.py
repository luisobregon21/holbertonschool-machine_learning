#!/usr/bin/env python3
''' Swapi API '''
import requests


def sentientPlanets():
    '''
    returns the list of names of the home planets
    of all sentient species.
    '''
    url = "https://swapi.dev/api/species/"
    planets = []
    while url is not None:
        response = requests.get(url)
        data = response.json()
        for species in data['results']:
            if species['designation'] == "sentient":
                if species['homeworld'] is not None:
                    planet_url = species['homeworld']
                    planet_data = requests.get(planet_url).json()
                    planets.append(planet_data['name'])
        url = data['next']
    return planets
