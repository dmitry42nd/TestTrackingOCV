function rigid_body_kin
    pts = rodrigues(randn(3,1))*[-1 -1 -1;
           -1 -1 1;
           -1 1 -1;
           -1 1 1;
           1 -1 -1;
           1 -1 1;
            1 1 -1;
           1 1 1]' + repmat([1;0;4],1,8);
   Rv = rodrigues([0.001; 0; 0]);     
   TimeSpan = 100;
   v_object= [0; 0; 0.1];
   a_object = [0.00; 0.00; 0.001];
   v_camera = [0; 0.009; 0.01];
   a_camera = [0;0.01;-0.01];
   
   
   
   v_object_curr = v_object;
   v_camera_curr = v_camera;
   
   cur_pts = pts;
   n = size(pts, 2);
   for j = 1:n
       pth{j} = [];
       pt_projs{j} = [];
   end
   campos  = zeros(3,1);
   for mom = 1:TimeSpan
       v_object_curr = v_object_curr + a_object;
       v_camera_curr = v_camera_curr + a_camera;
       campos = campos + v_camera_curr;
       cur_pts(:, 1) = Rv*cur_pts(:, 1) + v_object_curr;
       cur_pts_cam(:, 1) = cur_pts(:, 1) - campos;
       for j = 2:n
           vj = v_object_curr;% + omega*cross(k, pts(:, j)-pts(:, 1));
           cur_pts(:, j) = Rv*cur_pts(:, j) + vj;
           cur_pts_cam(:, j) = cur_pts(:, j) - campos;
       end
       for j = 1:n
           proj = cur_pts_cam(1:2, j)/cur_pts_cam(3, j) + 0.001*randn(2,1);
           pth{j} = [pth{j} cur_pts_cam(:, j)];
           pt_projs{j} = [pt_projs{j} proj];
       end
       ProjN = [];
       for i = 1:n           
           ProjN = [ProjN pt_projs{i}(:, end)];
       end
%        pts0 = pts - repmat(mean(pts, 2), 1, n);
%        [R,t] = OPnP(pts, ProjN);
   end

   show_motion(pth);
   
   ProjN = [];
   Pos0 = [];
   sc_coef = 0.3;
   for i = 1:n
       Pos0 = [Pos0 sc_coef*pth{i}(:, 1)];        
       ProjN = [ProjN pt_projs{i}(:, end)];
   end
   
   %i suppose i reconstructed 3d object points with wrong scale, so I scale my points by some coefficient sc_coef
   [R,t] = OPnP(sc_coef*pts, ProjN);
   %I estimate the velocit of object as object motion in global frame
   %divided by number of observations
   vest = t/TimeSpan;
   
   projs_all = [];
   for i = 1:TimeSpan
       projs_apt = [];
       for j = 1:n
           projs_apt = [projs_apt pt_projs{j}(:, i)];
       end
       projs_all = [projs_all; projs_apt];       
   end
   
   %optimization params are rotation, 
   p0 = [1; vest; zeros(3,1)];
   
   fun = @(p) mot_fun(p, sc_coef*pts, TimeSpan, projs_all(:), v_camera, a_camera, Rv);
   options = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt'); 

   pfin = lsqnonlin(fun, p0, [], [], options);
   
   abs(1/sc_coef - pfin(1))/(1/sc_coef)
end

function r = mot_fun(p, pts, T, projs, v_camera_curr0, a_camera, Rv)
    sc = p(1);
    v_object_curr = p(2:4);    
    projs_all = [];
    n = size(pts, 2);
    campos = zeros(3,1);
    v_camera_curr = v_camera_curr0;
    a_object = zeros(3,1);%p(5:7);
    Rv = rodrigues(p(5:7));
    ptsN = sc*pts;
    cur_pts = ptsN;
    for i = 1:T
       v_object_curr = v_object_curr + a_object;
       v_camera_curr = v_camera_curr + a_camera;
       campos = campos + v_camera_curr;
       cur_pts(:, 1) = Rv*cur_pts(:, 1) + v_object_curr;
       cur_pts_cam(:, 1) = cur_pts(:, 1) - campos;
       for j = 2:n
           vj = v_object_curr;% + omega*cross(k, pts(:, j)-pts(:, 1));
           cur_pts(:, j) = Rv*cur_pts(:, j) + vj;
           cur_pts_cam(:, j) = cur_pts(:, j) - campos;
       end       
        projs_pred = cur_pts_cam(1:2, :)./ repmat(cur_pts_cam(3, :), 2, 1);
        projs_all = [projs_all; projs_pred];
    end    
    r = projs - projs_all(:);
end
function show_motion(pth)
   close all;
   figure(1);
   hold on;
   plot(pth{1}(1,:), pth{1}(3,:), '*', 'Color', 'g');
   plot(pth{2}(1,:), pth{2}(3,:), '*', 'Color', 'b');
   plot(pth{5}(1,:), pth{4}(3,:), '*', 'Color', 'r');
end