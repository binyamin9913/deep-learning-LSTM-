clc
clear all
close all
% to show the data on the scatter graph
inputData=xlsread('DataSet.csv',1,'B1:E890')';
mu=zeros(size(inputData,1),1)
sig=zeros(size(inputData,1),1)
dataTrainStandardized=zeros(size(inputData,1),size(inputData,2))
for i=1:4
    mu(i) = mean(inputData(i,:));
    sig(i) = std(inputData(i,:));
    dataTrainStandardized(i,:) = (inputData(i,:) - mu(i)) / sig(i);
end

testDataInput=xlsread('DataSet.csv',1,'B1:E890')';
output=xlsread('DataSet.csv',1,'G1:G890')';
%call data 
numFeatures = 4;
numResponses = 1;
numHiddenUnits = 50;
%define data size
layers = [sequenceInputLayer(numFeatures),lstmLayer(numHiddenUnits),fullyConnectedLayer(numResponses),regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01);

net=trainNetwork(dataTrainStandardized,output,layers,options);

% to prediction
for i =1:4
    dataTrainStandardized(i,:) = (testDataInput(i,:) - mu(i)) / sig(i);
end
outputPrediction=predict(net,dataTrainStandardized)';       


