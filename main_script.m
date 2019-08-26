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
param.isRecord = false;
param.figRecord = [1];
param.loadAPIMap = false;
% param.plotScale = 0.000002; % google-map
param.plotScale = 1; % naver-map

env = Environment(vo, pkg, param);
env.runMethod = @vo.run;

env.run();
env.delete();