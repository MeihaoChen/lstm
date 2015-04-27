#!/bin/bash
module load torch-deps/7
luajit main.lua -mode 'evaluate'
