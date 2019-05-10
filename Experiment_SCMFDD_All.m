function Experiment_SCMFDD_All
%***************************************************************************
%set parameter(SCMFDD has 3 parameters,lamda,Mu and percentage)
%***************************************************************************
seed=1;
rand('state',seed);
CV=5;
lamda=2^2;
Mu=2^0;
percentage=0.45;
cross_validation(seed,CV,lamda,Mu,percentage);
end

%***************************************************************************
% 5 cross validation experiment
%***************************************************************************
function result=cross_validation(seed,CV,lamda,Mu,percentage)
load('SCMFDD_Dataset.mat')

interaction_matrix=drug_disease_association_matrix;

drug_similarity_matrix_list{1}=get_Jaccard_similarity(structure_feature_matrix);
drug_similarity_matrix_list{2}=get_Jaccard_similarity(target_feature_matrix);
drug_similarity_matrix_list{3}=get_Jaccard_similarity(pathway_feature_matrix);
drug_similarity_matrix_list{4}=get_Jaccard_similarity(enzyme_feature_matrix);
drug_similarity_matrix_list{5}=get_Jaccard_similarity(drug_drug_interaction_feature_matrix);

dis_similairty_matrix=normalized_dis_similairty_matrix;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[row,col]=size(interaction_matrix);
dimension_of_latent_vector=fix(percentage*min(row,col));%get the real number of latent according to the percentage

[row_index,col_index]=find(interaction_matrix==1);
link_num=sum(sum(interaction_matrix));
rand('state',seed);
random_index=randperm(link_num);
size_of_CV=round(link_num/CV);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result=zeros(5,7);      
for k=1:CV
    fprintf('begin to implement the cross validation:round =%d/%d\n', k, CV);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (k~=CV)
        test_row_index=row_index(random_index((size_of_CV*(k-1)+1):(size_of_CV*k)));
        test_col_index=col_index(random_index((size_of_CV*(k-1)+1):(size_of_CV*k)));
    else
        test_row_index=row_index(random_index((size_of_CV*(k-1)+1):end));
        test_col_index=col_index(random_index((size_of_CV*(k-1)+1):end));
    end

    train_interaction_matrix=interaction_matrix;
    test_link_num=size(test_row_index,1);
    for i=1:test_link_num
        train_interaction_matrix(test_row_index(i),test_col_index(i))=0;
    end
    
    
%*******************************************************************************
% call the method of MF_with_similarity_constraint to calculate the score matrix
%*******************************************************************************   
    for p=1:5
        drug_similarity_matrix=drug_similarity_matrix_list{p};
        predict_matrix_MF_constraint1=MF_with_similarity_constraint(drug_similarity_matrix,dis_similairty_matrix,train_interaction_matrix,lamda,Mu,dimension_of_latent_vector);
        predict_matrix_MF_constraint2=MF_with_similarity_constraint(dis_similairty_matrix,drug_similarity_matrix,train_interaction_matrix',lamda,Mu,dimension_of_latent_vector);
        predict_matrix_MF_constraint{p}=predict_matrix_MF_constraint1+predict_matrix_MF_constraint2';
    end
%*******************************************************************************
% Evaluation
%*******************************************************************************     
    for q=1:5
        result(q,:)=result(q,:)+model_evaluate(interaction_matrix,predict_matrix_MF_constraint{q},train_interaction_matrix); 
    end
    result/k    
end
result=result/CV;
end


function result=model_evaluate(interaction_matrix,predict_matrix,train_ddi_matrix)
real_score=interaction_matrix(:);
predict_score=predict_matrix(:);
index=train_ddi_matrix(:);
test_index=find(index==0);
real_score=real_score(test_index);
predict_score=predict_score(test_index);
aupr=AUPR(real_score,predict_score);
auc=AUC(real_score,predict_score);
[sen,spec,precision,accuracy,f1]=evaluation_metric(real_score,predict_score);
result=[aupr,auc,sen,spec,precision,accuracy,f1];
end

function [sen,spec,precision,accuracy,f1]=evaluation_metric(interaction_score,predict_score)
sort_predict_score=unique(sort(predict_score));
score_num=size(sort_predict_score,1);
threshold=sort_predict_score(ceil(score_num*(1:999)/1000));

for i=1:999
    predict_label=(predict_score>threshold(i));
    [temp_sen(i),temp_spec(i),temp_precision(i),temp_accuracy(i),temp_f1(i)]=classification_metric(interaction_score,predict_label);
end
[max_score,index]=max(temp_f1);
sen=temp_sen(index);
spec=temp_spec(index);
precision=temp_precision(index);
accuracy=temp_accuracy(index);
f1=temp_f1(index);
end

function [sen,spec,precision,accuracy,f1]=classification_metric(real_label,predict_label)
tp_index=find(real_label==1 & predict_label==1);
tp=size(tp_index,1);

tn_index=find(real_label==0 & predict_label==0);
tn=size(tn_index,1);

fp_index=find(real_label==0 & predict_label==1);
fp=size(fp_index,1);

fn_index=find(real_label==1 & predict_label==0);
fn=size(fn_index,1);

accuracy=(tn+tp)/(tn+tp+fn+fp);
sen=tp/(tp+fn);
recall=sen;
spec=tn/(tn+fp);
precision=tp/(tp+fp);
f1=2*recall*precision/(recall+precision);
end

function area=AUPR(real,predict)
sort_predict_score=unique(sort(predict));
score_num=size(sort_predict_score,1);
threshold=sort_predict_score(ceil(score_num*(1:999)/1000));

threshold=threshold';
threshold_num=length(threshold);
tn=zeros(threshold_num,1);
tp=zeros(threshold_num,1);
fn=zeros(threshold_num,1);
fp=zeros(threshold_num,1);

for i=1:threshold_num
    tp_index=logical(predict>=threshold(i) & real==1);
    tp(i,1)=sum(tp_index);
    
    tn_index=logical(predict<threshold(i) & real==0);
    tn(i,1)=sum(tn_index);
    
    fp_index=logical(predict>=threshold(i) & real==0);
    fp(i,1)=sum(fp_index);
    
    fn_index=logical(predict<threshold(i) & real==1);
    fn(i,1)=sum(fn_index);
end

sen=tp./(tp+fn);
precision=tp./(tp+fp);
recall=sen;
x=recall;
y=precision;
[x,index]=sort(x);
y=y(index,:);

area=0;
x(1,1)=0;
y(1,1)=1;
x(threshold_num+1,1)=1;
y(threshold_num+1,1)=0;
area=0.5*x(1)*(1+y(1));
for i=1:threshold_num
    area=area+(y(i)+y(i+1))*(x(i+1)-x(i))/2;
end
plot(x,y)
end

function area=AUC(real,predict)
sort_predict_score=unique(sort(predict));
score_num=size(sort_predict_score,1);
threshold=sort_predict_score(ceil(score_num*(1:999)/1000));

threshold=threshold';
threshold_num=length(threshold);
tn=zeros(threshold_num,1);
tp=zeros(threshold_num,1);
fn=zeros(threshold_num,1);
fp=zeros(threshold_num,1);
for i=1:threshold_num
    tp_index=logical(predict>=threshold(i) & real==1);
    tp(i,1)=sum(tp_index);
    
    tn_index=logical(predict<threshold(i) & real==0);
    tn(i,1)=sum(tn_index);
    
    fp_index=logical(predict>=threshold(i) & real==0);
    fp(i,1)=sum(fp_index);
    
    fn_index=logical(predict<threshold(i) & real==1);
    fn(i,1)=sum(fn_index);
end

sen=tp./(tp+fn);
spe=tn./(tn+fp);
y=sen;
x=1-spe;
[x,index]=sort(x);
y=y(index,:);
[y,index]=sort(y);
x=x(index,:);

area=0;
x(threshold_num+1,1)=1;
y(threshold_num+1,1)=1;
area=0.5*x(1)*y(1);
for i=1:threshold_num
    area=area+(y(i)+y(i+1))*(x(i+1)-x(i))/2;
end
plot(x,y)
end

%***************************************************************************
%normalized function
%***************************************************************************
function similarity_matrix=matrix_normalize(similarity_matrix)
similarity_matrix(isnan(similarity_matrix))=0;
[row,col]=size(similarity_matrix);
for i=1:row
    similarity_matrix(i,i)=0;
end
if row==col
    similarity_matrix(isnan(similarity_matrix))=0;
    for i=1:size(similarity_matrix,1)
        similarity_matrix(i,i)=0;
    end
    for round=1:200
        D=diag(sum(similarity_matrix,2));
        D1=pinv(sqrt(D));
        similarity_matrix=D1*similarity_matrix*D1;
    end
else
    for j=1:size(similarity_matrix,1)
        if sum( similarity_matrix(j,:))~=0
            similarity_matrix(j,:) = similarity_matrix(j,:)./ sum(similarity_matrix(j,:)); %bsxfun(@rdivide,W1,sum(W1,2));
        else
            similarity_matrix(j,:)=zeros(1,size(similarity_matrix,2));
        end
    end
end
end

%***************************************************************************
%Similarity Constraint Matrix Factorization Function
%Parameter:
%W_r:drug-drug similarity matrix, W_c:disease-disease similarity matrix
%Y:drug-disease association matrix
%***************************************************************************
function F=MF_with_similarity_constraint(W_r,W_c,Y,alpha,lamda,dimension_of_latent_vector) 
[row,col]=size(Y);
f=dimension_of_latent_vector;
seed=1;
rand('state',seed);
A=rand(row,f);
B=rand(col,f);
for k=1:50
    for i=1:row
        % By using the newer A(1,:),...,A(i-1), we can accelerate the iteration process, which is similar to Gauss-Seidel method
        A(i,:)=(Y(i,:)*B+alpha*(W_r(i,:)+(W_r(:,i))')*A)/(B'*B+alpha*sum(W_r(i,:)+(W_r(:,i))')*eye(f)+lamda*eye(f));
    end
    for j=1:col
        B(j,:)=(Y(:,j)'*A+alpha*(W_c(j,:)+(W_c(:,j))')*B)/(A'*A+alpha*sum(W_c(j,:)+(W_c(:,j))')*eye(f)+lamda*eye(f));
    end
end
F=A*B';
end

%***************************************************************************
%Jaccard Similarity Calculation
%***************************************************************************
function similarity_matrix = get_Jaccard_similarity(interaction_matrix)
    %get intersection matrix
    intersection_matrix = interaction_matrix * interaction_matrix';
    %get union matrix
    row_neighbor_num = sum(interaction_matrix, 2);
    row_matrix = row_neighbor_num * ones(1, size(row_neighbor_num, 1));
    col_matrix = row_matrix';
    union_matrix = col_matrix + row_matrix - intersection_matrix;
    %calculate similarity_matrix of row
    similarity_matrix = intersection_matrix ./ union_matrix;
    similarity_matrix(isnan(similarity_matrix)) = 0;
    for i = 1 : size(interaction_matrix, 1)
        similarity_matrix(i, i) = 0;
    end
    similarity_matrix = matrix_normalize(similarity_matrix);
end

