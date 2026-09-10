%% generate_mat_files.m
% Parse gain dumps into parameter/<name>.mat files for main2.m to load.
% Paste the MATLAB-formatted blocks printed by scripts/pth_reader.py into
% raw_data (filename comment, then Kp_lin ... Ki_rot). Run this script
% from deploy/sim_to_sim/. The repository ships parameter/ empty.

clear; clc;

% Create the parameter folder if it does not exist.
output_folder = 'parameter';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Paste gain dumps into this cell array.
% Each line is a quoted string; entries are comma-separated.
raw_data = {
        


%%% Predictor main lambda=0.25 beta=1.0 batch=256
'% Predictor_Seed_1_batch_256_lambda_0.25_beta_1.0'
'Kp_lin = [0.1785; 0.1785; 1.9391];'
'Ki_lin = [0.0007; 0.0007; 0.2291];'
'Kd_lin = [0.1914; 0.1914; 0.6228];'
'Kr_rot = [70000.00; 70000.00; 60000.00];'
'Kw_rot = [20000.00; 20000.00; 12000.00];'
'Ki_rot = [-4.4154; -4.4154; 506.9823];'

'% Predictor_Seed_42_batch_256_lambda_0.25_beta_1.0'
'Kp_lin = [0.1688; 0.1688; 1.4604];'
'Ki_lin = [0.0075; 0.0075; 0.2500];'
'Kd_lin = [0.2826; 0.2826; 0.5405];'
'Kr_rot = [70000.00; 70000.00; 60000.00];'
'Kw_rot = [20000.00; 20000.00; 12000.00];'
'Ki_rot = [-5.0000; -5.0000; 511.1072];'

'% Predictor_Seed_123_batch_256_lambda_0.25_beta_1.0'
'Kp_lin = [0.2020; 0.2020; 2.3473];'
'Ki_lin = [0.0084; 0.0084; 0.2500];'
'Kd_lin = [0.2034; 0.2034; 0.6993];'
'Kr_rot = [70000.00; 70000.00; 60000.00];'
'Kw_rot = [20000.00; 20000.00; 12000.00];'
'Ki_rot = [-3.3311; -3.3311; 504.7868];'

'% Predictor_Seed_456_batch_256_lambda_0.25_beta_1.0'
'Kp_lin = [0.2479; 0.2479; 2.2161];'
'Ki_lin = [0.1638; 0.1638; 0.2458];'
'Kd_lin = [0.3546; 0.3546; 0.7350];'
'Kr_rot = [70000.00; 70000.00; 60000.00];'
'Kw_rot = [20000.00; 20000.00; 12000.00];'
'Ki_rot = [-0.3578; -0.3578; 500.9348];'

'% Predictor_Seed_789_batch_256_lambda_0.25_beta_1.0'
'Kp_lin = [0.2662; 0.2662; 3.1925];'
'Ki_lin = [0.2661; 0.2661; 0.2500];'
'Kd_lin = [0.2430; 0.2430; 0.7992];'
'Kr_rot = [70000.00; 70000.00; 60000.00];'
'Kw_rot = [20000.00; 20000.00; 12000.00];'
'Ki_rot = [-4.4010; -4.4010; 498.7159];'



    
};

%% Parse the cell array into per-run .mat files.
current_filename = '';
vars_to_save = struct();

for i = 1:length(raw_data)
    line = strtrim(raw_data{i});
    
    % Skip empty lines and annotation-only rows (task logs, old dumps, section banners).
    if isempty(line) || contains(line, 'task') || contains(line, 'old:') || contains(line, '%%%')
        continue;
    end
    
    % 1. Filename: a comment line starting with % and containing no '='.
    if startsWith(line, '%') && ~contains(line, '=')
        % Flush the previous run before starting a new one.
        if ~isempty(current_filename) && isfield(vars_to_save, 'Kp_lin')
            save_mat(current_filename, vars_to_save, output_folder);
            vars_to_save = struct(); % reset for the next run
        end
        
        % Strip the leading % and trailing "&&& ..." notes.
        name_str = regexprep(line, '^%\s*', '');
        name_str = regexprep(name_str, '\s*&&&.*', ''); % drop "&&& mid" and similar
        current_filename = strtrim(name_str);
        continue;
    end
    
    % 2. Parameter assignment: extract the numeric vector.
    if contains(line, '=')
        % Drop a leading % if the assignment was commented out.
        line = regexprep(line, '^%\s*', '');
        % Drop trailing comments.
        line = regexprep(line, '%.*$', '');
        % Drop semicolons.
        line = strrep(line, ';', '');
        
        % Split "name = values".
        parts = split(line, '=');
        var_name = strtrim(parts{1});
        val_str = strtrim(parts{2});
        
        % eval turns '[0.1; 0.2; 0.3]' into a numeric column.
        try
            vars_to_save.(var_name) = eval(val_str);
        catch
            warning('Could not parse parameter line: %s', line);
        end
    end
end

% Save the last run.
if ~isempty(current_filename) && isfield(vars_to_save, 'Kp_lin')
    save_mat(current_filename, vars_to_save, output_folder);
end

fprintf('Finished writing .mat files. Check the %s folder.\n', output_folder);

%% Helper: write one .mat file from a struct of gains.
function save_mat(filename, data_struct, out_dir)
    % Require a complete gain set before saving.
    if isfield(data_struct, 'Kp_lin') && isfield(data_struct, 'Ki_rot')
        % Promote struct fields into the function workspace for save().
        Kp_lin = data_struct.Kp_lin;
        Ki_lin = data_struct.Ki_lin;
        Kd_lin = data_struct.Kd_lin;
        Kr_rot = data_struct.Kr_rot;
        Kw_rot = data_struct.Kw_rot;
        Ki_rot = data_struct.Ki_rot;
        
        filepath = fullfile(out_dir, [filename, '.mat']);
        save(filepath, 'Kp_lin', 'Ki_lin', 'Kd_lin', 'Kr_rot', 'Kw_rot', 'Ki_rot');
        fprintf('Wrote: %s.mat\n', filename);
    else
        warning('Incomplete data for %s; skip save.', filename);
    end
end
