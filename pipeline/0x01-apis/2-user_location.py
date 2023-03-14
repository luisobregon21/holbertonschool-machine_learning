#!/usr/bin/env python3
''' prints the location of a specific user '''
import sys
import requests
import datetime

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} <API_URL>".format(sys.argv[0]))
        sys.exit(1)

    api_url = sys.argv[1]
    response = requests.get(api_url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_time = datetime.datetime.fromtimestamp(
            int(response.headers['X-Ratelimit-Reset']))
        time_diff = reset_time - datetime.datetime.now()
        print("Reset in {} min".format(int(time_diff.total_seconds() / 60)))
    elif response.status_code == 200:
        user_data = response.json()
        if user_data['location'] is not None:
            print(user_data['location'])
        else:
            print("No location provided")
    else:
        print("Error: {}".format(response.status_code))
