%%% This is a script of adaboost of metrics
clear;clc;
imgDir='../image/';
addpath('../');

addpath(genpath('Assistant Code'));
%%%%%%
num_person=632;
Partition;
%%%%%load different features
LoadFeatures;

