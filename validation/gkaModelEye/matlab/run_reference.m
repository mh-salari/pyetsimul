function run_reference()
% Generate the gkaModelEye reference outputs for the validation scene; writes ../reference.json.
% Run with gkaModelEye on the path (https://github.com/gkaguirrelab/gkaModelEye).

    here = fileparts(mfilename('fullpath'));

    % Lab scene (PyEtSimul mm frame: x right, y from the screen toward the participant, z up).
    eye = [30; 835; 85];
    cam = [-100; 420; -125];
    light = [95; 435; -115];
    screen_w = 531.36;
    screen_h = 298.98;
    cal_half_w = 0.88 * screen_w / 2;
    cal_half_h = 0.83 * screen_h / 2;
    targets = [0, 0, 0; cal_half_w, 0, 0; -cal_half_w, 0, 0; 0, 0, cal_half_h; 0, 0, -cal_half_h];
    stop_radius_mm = 3.0;  % aperture stop (dilated; about a 6 mm pupil)

    % gkaModelEye places the eye at the origin with axes p1 (forward), p2 (horizontal), p3 (vertical).
    % Forward is toward the screen (-y in the lab frame), up is +z, p2 = p3 x p1.
    p1 = [0, -1, 0];
    p3 = [0, 0, 1];
    p2 = cross(p3, p1);
    frame = [p1; p2; p3];
    cam_ef = frame * (cam - eye);
    light_ef = frame * (light - eye);
    camera_translation = [cam_ef(2); cam_ef(3); cam_ef(1)];  % [horizontal; vertical; depth]
    glint_relative = [light_ef(2) - cam_ef(2); light_ef(3) - cam_ef(3); light_ef(1) - cam_ef(1)];

    sg = createSceneGeometry('spectralDomain', 'nir', ...
        'cameraTranslation', camera_translation, ...
        'cameraGlintSourceRelative', glint_relative);

    results = struct('target_mm', {}, 'pupil_center_px', {}, 'glint_px', {});
    for i = 1:size(targets, 1)
        gaze = targets(i, :)' - eye;
        gaze = gaze / norm(gaze);
        gaze_ef = frame * gaze;
        azimuth = atan2d(gaze_ef(2), gaze_ef(1));
        elevation = asind(gaze_ef(3));

        [pupil_ellipse, glint] = projectModelEye([azimuth, elevation, 0, stop_radius_mm], sg);

        results(i).target_mm = targets(i, :);
        results(i).pupil_center_px = pupil_ellipse(1:2);
        results(i).glint_px = glint(:)';
    end

    out.scene = struct( ...
        'eye_position_mm', eye', ...
        'camera_position_mm', cam', ...
        'light_position_mm', light', ...
        'stop_radius_mm', stop_radius_mm, ...
        'intrinsic_matrix', sg.cameraIntrinsic.matrix);
    out.results = results;

    fid = fopen(fullfile(here, '..', 'reference.json'), 'w');
    fwrite(fid, jsonencode(out, 'PrettyPrint', true));
    fclose(fid);
    fprintf('Wrote reference.json (%d targets)\n', size(targets, 1));
end
