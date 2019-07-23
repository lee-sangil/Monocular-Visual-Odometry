clc
clear
close all
clearvars -global

addpath(genpath('../5-point-algorithm-MATLAB-master/'));
addpath(genpath('utils'));

profile off
profile on

%% PACKAGE CLASS
pkg = HYUNDAI('D:\Libraries\Documents\OneDrive - SNU\Doing\# Project\현대엠엔소프트\데이터셋\2019_0603_sample\');
pkg.imInit = 1000;
pkg.imLength = 50;

% pkg = KITTI(8);
% pkg = VIRTUAL();
read(pkg);

%% VO CLASS
vo = vo_mono(pkg);

%% SCRIPT
params.isRecord = true;
params.figRecord = [1];

env = Environment(vo, pkg, params);
env.runMethod = @vo.run;

env.run();
env.delete();