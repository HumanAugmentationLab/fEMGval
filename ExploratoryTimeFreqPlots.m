clear

% To do
% - make colorbar scales the same and centered on zero with better colors
% - imagesc time series plots might not have right time (comparison plots)

% Load the epocheddata.
%load data/smilefrown1filt0p5notch56t64epochs.mat
%load data/smilefrown2filt0p5notch56t64epochs.mat
%load data/smilefrown2filt0p5notch56t64p120epoch.mat
%load data/run15filt0p5doublenotch56t64a120epochs.mat
%load data/run16rawdatafilt0p5notch56t64epochs.mat
%load data/vivSF2.mat
load data/run17rawdatafilt0p5notch56t64epochs.mat

%load data/smilefrownangryblinkS1filt0p5notch56t64epochs.mat

EEG.timessec = EEG.times./1000; %version of times in seconds, useful for signal processing

load data/colormapjetwhite.mat; %loads cmapwj to do a colormap with white instead of green for jet


% Create a new categorical variable for easier manipulation.
% Note, this will be inaccurate if you select a subset of the data without
% subsetting this.
for epoch = 1:EEG.trials
    for i = 1:length(EEG.epoch(epoch).eventlatency)
        if EEG.epoch(epoch).eventlatency{i} == 0
            EEG.epochlabelscat{epoch} = EEG.epoch(epoch).eventtype{i};
        end
    end
end
EEG.epochlabelscat = categorical(EEG.epochlabelscat);
%make an array of all possible labels of events in your data
availableeventlabels = unique(EEG.epochlabelscat)
%We'll use this to index which epochs we want

%% Do any additional filtering or trial baseline removal that you want 
% (currently another high pass at 20 hz) and also find power first

origEEGdata = EEG.data;
for channel =1:size(EEG.data,1)
    % First, save the power in the frequency domain
    [EEG.freqcalcs.P(channel,:,:),EEG.freqcalcs.F] = pspectrum(squeeze(EEG.data(channel,:,:)),EEG.timessec);
    
    
    %Trial by trial baseline removal (can also do pre-event baseline removal
    %instead
    EEG.data(channel,:, :) = EEG.data(channel,:, :) - mean(EEG.data(channel,:, :),2);
    
    % additional filtering (might want to do this with the continuous data
    % instead, if you find somethign you like)
    EEG.data(channel,:, :) = highpass(squeeze(EEG.data(channel,:, :)),20,EEG.srate);
end

%% Settings for what plots you want to make
% Now we'll do some calculations to either compare two conditions (or just do 
% a singular one) [could improve this by taking more than 2]. We'll store this 
% in some data structures.

condnames = {"SPACE pressed", "DOWN pressed"};
channels = 1;

% Compare the time series using imagesc and then make a line plot with both
% time series and error bars
optionsforplot.comparativetimeseries = true; 
optionsforplot.commonvoltageaxis = true; %use the same color scale for the imagesc plots
optionsforplot.caxisvolt = [-10 10];

optionsforplot.comparativewavelet = true;
optionsforplot.commonpoweraxis = true;
optionsforplot.caxispower = [0 2.5];
optionsforplot.differencecond = true;
optionsforplot.caxispowerdiff = [-1.5 1.5];





%% Run some analysis to compute general things
clear condstruc idxtrials 
for c = 1:length(condnames)
    condname = condnames{c}
    idxtrials = find(EEG.epochlabelscat==condname);
    
    for ch = 1:length(channels)
        channel = channels(ch); % in case the channel order isn't linear
        
        % Mean of timecourse
        condstruc(c,ch).meantimecourse = mean(squeeze(EEG.data(channel,:, idxtrials)),2); 
        condstruc(c,ch).stdtimecourse = std(squeeze(EEG.data(channel,:,idxtrials)),0,2);  
        condstruc(c,ch).semtimecourse = condstruc(c,ch).stdtimecourse./sqrt(length(idxtrials)); % Compute 'Standard Error Of The Mean'
        condstruc(c,ch).CI95 = tinv([0.025 0.975], length(idxtrials)-1); % Calculate 95% Probability Intervals Of t-Distribution (might do z intead, largely in consequential)
        condstruc(c,ch).yCI95timecourse = bsxfun(@times, condstruc(c,ch).semtimecourse, condstruc(c,ch).CI95(:)');              % Calculate 95% Confidence Intervals
        
        for i  = 1:length(idxtrials)
            [condstruc(c,ch).cfs(i,:,:),condstruc(c,ch).frq] = cwt(double(EEG.data(channel,:, idxtrials(i))),EEG.srate,'FrequencyLimits',[2 120]);
        end
       
        condstruc(c,ch).meancfs = squeeze(mean(abs(condstruc(c,ch).cfs),1)); %adding an abs here to get magnitude of a complex number
        condstruc(c,ch).stdcfs = squeeze(std(abs(condstruc(c,ch).cfs))); % may actually want to do the stdev across all time w/in a freq too
    end
end



%% Plot Comparative Time Series (individual and mean times series)
if optionsforplot.comparativetimeseries
    for ch = 1:length(channels) % you can replace these if you just want to plot some
        channel = channels(ch); % in case the channel order isn't linear

        figure; set(gcf,'Visible','on','WindowState','maximized') % This pops out the figure for easier viewing

        for c = 1:length(condnames)
            condname = condnames{c};
            idxtrials = find(EEG.epochlabelscat==condname);
            % Time series
            subplot(3,1,c)
            % Plot raw data
            imagesc(squeeze(EEG.data(channel,:, idxtrials))'); 
            set(gca, 'XTick', (EEG.xmin*1000):500:(EEG.xmax*1000));
            if optionsforplot.commonvoltageaxis; caxis(optionsforplot.caxisvolt);end
            colormap(cmapwj); colorbar;
            ylabel('Trial'); xlabel('Time'); 
            title(strcat('Time  Plots for: ', condname,' Ch: ', num2str(channel)));
        end
        % Time series compare
        subplot(3,1,3)
        plot(EEG.timessec, condstruc(1,ch).meantimecourse,'b','LineWidth',2);
        hold on
        plot(EEG.timessec, condstruc(2,ch).meantimecourse,'r','LineWidth',2);

        ciplotter(condstruc(1,ch).yCI95timecourse(:,1)+condstruc(1,ch).meantimecourse,condstruc(1,ch).yCI95timecourse(:,2)+condstruc(1,ch).meantimecourse,EEG.timessec,'b',.5);
        ciplotter(condstruc(2,ch).yCI95timecourse(:,1)+condstruc(2,ch).meantimecourse,condstruc(2,ch).yCI95timecourse(:,2)+condstruc(2,ch).meantimecourse,EEG.timessec,'r',.5);
        plot(EEG.timessec, condstruc(1,ch).meantimecourse,'b','LineWidth',2);
        plot(EEG.timessec, condstruc(2,ch).meantimecourse,'r','LineWidth',2);
        legend(condnames)
        ylabel('Voltage'); xlabel('Time (s)');
        title(strcat('Time Freq Plots for: ', condnames{1},' and ',condnames{2},' Ch: ', num2str(channel)));
    end
end
%% Plot Comparative Wavelet Analysis
if optionsforplot.comparativewavelet
   
    % Generate wavelet plot for each condition
    for ch = 1:length(channels) % you can replace these if you just want to plot some
        channel = channels(ch); % in case the channel order isn't linear
        figure
        for c = 1:length(condnames)
            condname = condnames{c};
            idxtrials = find(EEG.epochlabelscat==condname);
            
            %Continuous wavelet transform (average)
            if optionsforplot.differencecond; subplot(3,1,c); else; subplot(2,1,c); end
            colormap('copper');
            surface(EEG.timessec,condstruc(c,ch).frq,condstruc(c,ch).meancfs); % moved abs to before we take th mean
            if optionsforplot.commonpoweraxis; caxis(optionsforplot.caxispower); end
            colorbar;
            axis tight; shading flat;
            xlabel('Time (s)')
            ylabel('Frequency (Hz)')
            title(strcat('Time Freq Plots for: ', condname,' Ch: ', num2str(channel)));
            %set(gca,'yscale','log')    
        end
    end
        
    % Generate comparative (t-value) wavelet plot for 2-1
    
    if optionsforplot.differencecond && length(condnames)> 1  %This does plots for 2-1 so red goes w/ plots in subplot1
        clear a1 a2 a3 pooledstd zish
        a1 = ((size(condstruc(1,ch).cfs,1)-1) .* (condstruc(1,ch).stdcfs.^2)); % weighted condition 1
        a2 = ((size(condstruc(2,ch).cfs,1)-1) .* (condstruc(2,ch).stdcfs.^2)); % weighted condition 2
        a3 = (size(condstruc(1,ch).cfs,1)+size(condstruc(2,ch).cfs,1)-2); %combined denom
        pooledstd = sqrt((a1+a2)./a3);
        zish = (condstruc(2,ch).meancfs - condstruc(1,ch).meancfs)./(pooledstd);
        subplot(3,1,3)
        surface(EEG.timessec,condstruc(1,ch).frq,zish); % not sure if this is right log?
        colorbar; colormap(cmapwj); caxis(optionsforplot.caxispowerdiff); %set color of surface and colorbar/scheme axis
        axis tight; shading flat
        xlabel('Time (s)'); ylabel('Frequency (Hz)');
        title(strcat("Zval difference for ", condnames{2}," minus ", condnames{1}," Ch: ", num2str(channel)));
        %set(gca,'yscale','log')op
    end
end



    
