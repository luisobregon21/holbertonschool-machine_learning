#!/usr/bin/env python3
''' Swapi API '''
import requests


def availableShips(passengerCount):
    '''
    create a method that returns the list of ships that
    can hold a given number of passengers
    :passengerCount: number of passengers
    '''
    starships = []
    r = requests.get('https://swapi-api.hbtn.io/api/starships').json()
    while (r['next']):
        for ship in r['results']:
            if ship['passengers'].isdigit() is True:
                if int(ship['passengers']) >= passengerCount:
                    starships.append(ship['name'])
        r = requests.get(r['next']).json()

    return starships
