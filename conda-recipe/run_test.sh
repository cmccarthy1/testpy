#!/bin/bash
if [ -e ${QLIC}/k4.lic ]
then
  echo Tests using tox should be run here;
else
  echo No kdb+, no tests;
fi
