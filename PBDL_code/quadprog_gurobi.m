function [x,fval,exitflag,output,lambda] = quadprog_gurobi(H,f,A,b,Aeq,beq,lb,ub,options)
% this function is downloaded and a bit modified from Gurobi website. it is equavalent of
% quad prog in matlab. read quadprog documentation if you are interested.

% Build Gurobi model
model.Q = sparse(H);
model.obj = f;
model.A = [sparse(A); sparse(Aeq)]; % A must be sparse
model.sense = [repmat('<',size(A,1),1); repmat('=',size(Aeq,1),1)];
model.rhs = full([b(:); beq(:)]); % rhs must be dense
if ~isempty(lb)
    model.lb = lb;
else
    model.lb = -inf(size(model.A,2),1); % default lb for MATLAB is -inf
end
if ~isempty(ub)
    model.ub = ub;
end

% Extract relevant Gurobi parameters from (subset of) options
params = struct();

if isfield(options,'Display') || isa(options,'optim.options.SolverOptions')
    if any(strcmp(options.Display,{'off','none'}))
        params.OutputFlag = 0;
    end
end


% Solve model with Gurobi
result = gurobi(model,params);

% Resolve model if status is INF_OR_UNBD
if strcmp(result.status,'INF_OR_UNBD')
    params.DualReductions = 0;
    warning('Infeasible or unbounded, resolve without dual reductions to determine...');
    result = gurobi(model,params);
end

% Collect results
x = [];
output.message = result.status;
output.constrviolation = [];

if isfield(result,'x')
    x = result.x;
    if nargout > 3
        slack = model.A*x-model.rhs;
        violA = slack(1:size(A,1));
        violAeq = norm(slack((size(A,1)+1):end),inf);
        viollb = model.lb(:)-x;
        violub = 0;
        if isfield(model,'ub')
            violub = x-model.ub(:);
        end
        output.constrviolation = max([0; violA; violAeq; viollb; violub]);
    end
end

fval = [];

if isfield(result,'objval')
    fval = result.objval;
end

if strcmp(result.status,'OPTIMAL')
    exitflag = 1; % converged to a solution
elseif strcmp(result.status,'UNBOUNDED')
    exitflag = -3; % problem is unbounded
elseif strcmp(result.status,'ITERATION_LIMIT')
    exitflag = 0; % maximum number of iterations reached
else
    exitflag = -2; % no feasible point found
end

lambda.lower = [];
lambda.upper = [];
lambda.ineqlin = [];
lambda.eqlin = [];

if nargout > 4
    if isfield(result,'rc')
        lambda.lower = max(0,result.rc);
        lambda.upper = -min(0,result.rc);
    end
    if isfield(result,'pi')
        if ~isempty(A)
            lambda.ineqlin = -result.pi(1:size(A,1));
        end
        if ~isempty(Aeq)
            lambda.eqlin = -result.pi((size(A,1)+1):end);
        end
    end
end
