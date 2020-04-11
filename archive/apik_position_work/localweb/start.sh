#!/bin/sh

cd "$(dirname "$0")"
flask run --host 0.0.0.0 --port 80
