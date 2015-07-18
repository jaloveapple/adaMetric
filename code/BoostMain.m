%%% This is a script of adaboost of metrics
clear;clc;
imgDir='../image/';
addpath('../');

addpath(genpath('Assistant Code'));
%%%%%%
toysize=40;
Partition;
%%%%%load different features
LoadFeatures;

