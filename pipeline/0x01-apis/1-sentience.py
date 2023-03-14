#!/usr/bin/env python3
''' Swapi API '''
import requests


def sentientPlanets():
    '''
    returns the list of names of the home planets
    of all sentient species.
    '''
    planets = []
    url = 'https://swapi-api.hbtn.io/api/species/'
    while url is not None:
        response = requests.get(url,
                                headers={'Accept': 'application/json'},
                                params={"term": 'specie'})
        for specie in response.json()['results']:
            if specie['classification'] == 'sentient' or \
                    specie['designation'] == 'sentient':
                if specie['homeworld'] is not None:
                    homeworld = requests.get(specie['homeworld'])
                    planets.append(homeworld.json()['name'])

        url = response.json()['next']
    return planets
