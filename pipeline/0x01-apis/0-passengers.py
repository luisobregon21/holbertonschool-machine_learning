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
    url = 'https://swapi-api.hbtn.io/api/starships/'
    while url is not None:
        response = requests.get(url,
                                headers={'Accept': 'application/json'},
                                params={"term": 'starships'})
        for ship in response.json()['results']:
            passenger = ship['passengers'].replace(',', '')
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                starships.append(ship['name'])
        url = response.json()['next']
    return starships
