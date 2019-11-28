% 
% Stepinfo
% by Aidan Macdonald (address@hidden)
% 
% Written to match specification found on
% www.mathwords.com/help/control/ref/stepinfo.html
% 
% 


function S = stepinfo_Q1(sys);

[y, t] = step(sys);
RT = [0.1, 0.9];
ST = 0.02;
   
% Find max points
[ypeak, i] = max(y);
tpeak = t(i);

[h, w] = size(y);

% Divide the signal into initial and final segments
% divided on the max point
yhead = y(1:i); % Everything before the peak
ytail = y(i:h); % Everything after peak
ttail = t(i:h);

[hf, wf] = size(ytail);

% Estimate the final convergent value
% Get average of second half of ytail
yfinal = mean(ytail(max (floor (hf/2), 1):hf));

% Estimate Rise time
% Capture indeces and errors of when
% the yhead reaches RT of yfinal
[err_low,  ilow]  = min(abs (yhead - yfinal*RT(1)));
[err_high, ihigh] = min(abs (yhead - yfinal*RT(2)));

% Get slope of line through the signal values
slope = (t(ihigh) - t(ilow))/(y(ihigh) - y(ilow));
% Errors used with linear approx to attempt to remove
% errors from the time measures
tlow = t(ilow) + slope*err_low;
thigh = t(ihigh) + slope*err_high;

rise_time = thigh - tlow;
% End of Rise Time Calculations

% Calculate settling params
settling_max = max(ytail);
settling_min = min(y(ihigh:h));

% Overshoot defined as percentage shot over the final
overshoot = (ypeak - yfinal)/yfinal;
undershoot = (yfinal - min(ytail))/yfinal;
% End Overshoot Calc

% Calculate Settling Time
% Take average over 100 samples
yavg = conv(ytail,ones(1,100)/100);
yavg = yavg(1:length(ttail));

yfinal;
[ysettle, settling_ind] = min(abs(abs(yavg - yfinal) - ST*ypeak));
settling_time = ttail(settling_ind);

S = struct('RiseTime', rise_time,'SettlingTime', settling_time,'SettlingMin', settling_min,'SettlingMax', settling_max,'Overshoot', overshoot,'Undershoot', undershoot,'Peak', abs(ypeak),'PeakTime', tpeak);
rise_time
settling_time
overshoot

end