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
pkg.imLength = 10;

% pkg = KITTI(8);
% pkg = VIRTUAL();
read(pkg);

%% VO CLASS
vo = vo_mono(pkg);

%% SCRIPT
params.isRecord = true;
params.figRecord = [1];
params.plotScale = .5;

env = Environment(vo, pkg, params);
env.runMethod = @vo.run;

env.run();
env.delete();