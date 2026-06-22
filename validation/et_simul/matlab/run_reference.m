function run_reference()
% Generate the et_simul reference outputs for the et_simul validation scene; writes ../reference.json.
% Run with the et_simul MATLAB on the path (https://github.com/mh-salari/et_simul-1.01).

    here = fileparts(mfilename('fullpath'));

    % et_simul applies angle kappa in eye_look_at only when this is set.
    global FEAT_FOVEA_DISPLACEMENT %#ok<GVMIS>
    FEAT_FOVEA_DISPLACEMENT = true;

    mm = 1e-3;

    % Lab scene (PyEtSimul mm frame: x right, y from the screen toward the participant, z up).
    eye_pos = [30; 835; 85];
    cam_pos = [-100; 420; -125];
    light_pos = [95; 435; -115];
    pupil_diameter_mm = 6.0;

    % Gaze sweep on the screen plane (y = 0): centre + four cardinal midpoints of the calibration area.
    screen_w = 531.36;
    screen_h = 298.98;
    cal_half_w = 0.88 * screen_w / 2;
    cal_half_h = 0.83 * screen_h / 2;
    targets = [0, 0, 0; 0, 0, cal_half_h; cal_half_w, 0, 0; 0, 0, -cal_half_h; -cal_half_w, 0, 0];

    light = light_make();
    light.pos = [light_pos * mm; 1];
    camera = camera_make();
    camera.trans(1:3, 4) = cam_pos * mm;
    camera = camera_point_at(camera, eye_pos * mm);

    results = struct('target_mm', {}, 'pupil_center_px', {}, 'glint_px', {});
    for i = 1:size(targets, 1)
        target = targets(i, :)';

        eye = eye_make(7.98e-3);
        eye.trans(1:3, 4) = eye_pos * mm;
        eye = eye_look_at(eye, target * mm);

        pupil_img = eye_get_pupil_image(eye, camera);
        ellipse = fitellipse_hf(pupil_img(1, :)', pupil_img(2, :)');
        pupil_center = ellipse(1:2);

        cr = eye_find_cr(eye, light, camera);
        [glint, ~, valid] = camera_project(camera, cr);
        if ~valid
            glint = [NaN; NaN];
        end

        results(i).target_mm = target';
        results(i).pupil_center_px = pupil_center(:)';
        results(i).glint_px = glint(1:2)';
    end

    out.scene = struct( ...
        'eye_position_mm', eye_pos', ...
        'camera_position_mm', cam_pos', ...
        'light_position_mm', light_pos', ...
        'pupil_diameter_mm', pupil_diameter_mm);
    out.results = results;

    fid = fopen(fullfile(here, '..', 'reference.json'), 'w');
    fwrite(fid, jsonencode(out, 'PrettyPrint', true));
    fclose(fid);
    fprintf('Wrote reference.json (%d targets)\n', size(targets, 1));
end
