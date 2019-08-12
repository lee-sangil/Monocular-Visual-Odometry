clc
clear
close all
clearvars -global

addpath(genpath('utils'));

profile off
profile on

%% PACKAGE CLASS
pkg = HYUNDAI('D:\Libraries\Documents\OneDrive - SNU\Doing\# Project\현대엠엔소프트\데이터셋\2019_0603_sample\');
pkg.imInit = 900;
pkg.imLength = 1500;

% pkg = KITTI(8);
% pkg = VIRTUAL();
read(pkg);

%% VO CLASS
vo = vo_mono(pkg);

%% SCRIPT
params.isRecord = false;
params.figRecord = [1 2];
params.plotScale = 0.000002;

env = Environment(vo, pkg, params);
env.runMethod = @vo.run;

env.run();
env.delete();