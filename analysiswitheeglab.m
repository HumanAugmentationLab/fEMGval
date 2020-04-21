% Quick analysis of space bar with EEGLAB
a = pwd;
cd('C:\Users\saman\Documents\MATLAB\BCILAB\dependencies\eeglab2019_1')
eeglab
cd(a)
%%
EEG = pop_loadxdf(); % running with trial 10


%%
%% Generating markers to add in with the regular markers (based on completed lines and other manual counting)
% This one is specifically for run15, need to manually turn this into a
% text file in the future (type, latency (in seconds)
load('run15completedlines.mat');
markertype = CompletedLines(:,1);
smt = num2str(markertype);
type = cellstr(strcat('Completed',smt));
latency = CompletedLines(:,2);
eventtable = table(type,latency)

% Also include the game overs
timegameoverbaddrop = [ (8*60)+25, (11*60)+50, (14*60)+55,(2*60)+49, (7*60)+48]';
gameoverbaddrop = {'GameOver','GameOver','GameOver','BadDrop','BadDrop'};
for i =1:length(gameoverbaddrop)
    eventtable = [eventtable; {gameoverbaddrop{i}, timegameoverbaddrop(i)}];
end

%save('run15manualevents.mat','markertype','markertime');
writetable(eventtable,'run15manualevents.txt')
% then import event info from .txt file. Be sure to append
%% only emg channels
EEG = pop_select(EEG,'channel',1:2); %just select the 2 channels with data.

% high pass filter
EEG = pop_eegfiltnew(EEG,0.5); 

EEG = pop_eegfiltnew(EEG,56,64,'revfilt',1); %notch filter 

% This runs out of memory
%EEG = pop_cleanline(EEG, 'Bandwidth',2,'ChanCompIndices',[1:EEG.nbchan],                  ...
%                            'SignalType','Channels','ComputeSpectralPower',true,             ...
%                            'LineFrequencies',[60 120] ,'NormalizeSpectrum',false,           ...
%                            'LineAlpha',0.01,'PaddingFactor',2,'PlotFigures',false,          ...
%                            'ScanForLines',true,'SmoothingFactor',100,'VerboseOutput',1,    ...
%                            'SlidingWinLength',EEG.pnts/EEG.srate,'SlidingWinStep',4);


%% Epoch based on condition
EMGspace = pop_epoch(EEG, {'SPACE pressed'},[-.5 3.5]);
EMGright = pop_epoch(EEG, {'RIGHT pressed'},[-.5 3.5]);
EMGleft = pop_epoch(EEG, {'LEFT pressed'},[-.5 3.5]);
EMGdown = pop_epoch(EEG, {'DOWN pressed'},[-.5 3.5]);

namesEMG = {'EMGspace','EMGdown','EMGleft','EMGright'}
%%

pop_erpimage(EMGspace,1)
pop_erpimage(EMGdown,1)
pop_erpimage(EMGleft,1)
%% Plot each of the conditions 
for ch = 1:2
    figure
    for i = 1:length(namesEMG)
        subplot(2,ceil(length(namesEMG)./2),i)
        tempEMG = eval(namesEMG{i});
        erpimage( mean(tempEMG.data([ch], :),1), ones(1, tempEMG.trials)*tempEMG.xmax*1000, linspace(tempEMG.xmin*1000, tempEMG.xmax*1000, tempEMG.pnts),...
            strcat(namesEMG{i},' - Ch ',num2str(ch)), 10, 1 ,'yerplabel','\muV','erp','on','cbar','on');
    end
end


%% Remove baseline that is overall epoch
EMGspace = pop_rmbase(EMGspace,[],[],[]);
EMGright = pop_rmbase(EMGright,[],[],[]);
EMGleft = pop_rmbase(EMGleft,[],[],[]);
EMGdown = pop_rmbase(EMGdown,[],[],[]);

%%
ALLEMG = [EMGspace,EMGdown,EMGleft,EMGright];
%% Plot time series
%pop_comperp(ALLEEG,1,[1],[2]); % This version does the difference ERP

% Plot using CI plot
num_conditions = 2;
colors{1} = [.1 .3 .7];
colors{2} = [.7 .1 .2];
alphaval = .2;
ch = 1;
condstoplot = [2];

figure
for i = 1:length(condstoplot)
    tempEMG = eval(namesEMG{condstoplot(i)}); 
    x = tempEMG.times;
    y = squeeze(tempEMG.data(ch,:,:))';
    
    % Calcuate the confidence interval
    N = size(y,1);                                      % Number of ‘Experiments’ In Data Set
    yMean(i,:) = mean(y);                               % Mean Of All Experiments At Each Value Of ‘x’
    ySEM = std(y)/sqrt(N);                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
    CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
    yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’

    %plot(x,yMean(i,:),'Color',colors{i},'LineWidth',2);
    % Plot these both first.
    % Plot the confidence interval and mean
    curdir = pwd;
    cd('C:\Users\saman\Documents\MATLAB\HALBCI\ciplot\');
    ciplot((yCI95(1,:)+yMean(i,:)),(yCI95(2,:)+yMean(i,:)),x,colors{i},alphaval);
    hold on
    cd(curdir);
end
%
xline(0,'k--','LineWidth',2)
yline(0,'k--','LineWidth',2)

for i = 1:length(condstoplot)
    plot(x,yMean(i,:),'Color',colors{i},'LineWidth',2);
end

set(gca,'FontSize',14)
legend(namesEMG(condstoplot))
xlabel('Time (ms)');
ylabel('Signal (uV)');


%% Try aligning signals

% First for down

sys = ssest(squeeze(EMGdown.data(1,:,:)),3)



%% other stuff down here
%%
pop_saveset(EEG,'valve10.set'); %doing this because UI problems with loading .xdf (probably just a path thing.

%% load from .set file

EEG  = pop_loadset();

%% only emg channels
EEG = pop_select(EEG,'channel',1:2); %just select the 2 channels with data.

%% high pass filter
EEG = pop_eegfiltnew(EEG,0.1); 

% might want to notch filter with cleanline, but this might be done by the
% hardware, not seeing a lot of 60 hz.
%%
[Streams] = load_xdf('valve_03132020_trial15_openbci.xdf','Verbose','true');

%% Determine which streams have markers or data
mrkidx = [];
eegidx = [];
for i = 1:size(Streams,2)
    if strcmp('Markers',Streams{1,i}.info.type)
        mrkidx = [mrkidx i];
    elseif strcmp('EEG',Streams{1,i}.info.type)
        eegidx = [eegidx i];
    end
end

%%
EMG = Streams{1,eegidx};
plot(EMG.time_stamps,EMG.time_series(1,:))

%%
EMG = pop_select(EMG,'channel',1:2);
EMG = pop_eegfiltnew(EMG,0.1); 

%%
plot(EMG.time_stamps,EMG.time_series(1,:))




