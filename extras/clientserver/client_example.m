clc;
clear;

filename=input('Enter Filename:','s');

client = tcpclient("localhost",28567);

latency_matrix = zeros(50,101);
for rep = 1:50
    rep
    latency_vec = zeros(1,101);

    write(client,"new session","string");

    start = tic;
    response = read(client, client.NumBytesAvailable, "string");
    while isempty(response)
        response = read(client, client.NumBytesAvailable, "string");
    end
    latency = toc(start);
    latency_vec(1) = latency;

    for n = 1:100
        ar = randi([0 1], 1,2);
        write(client,sprintf("a,r:%d,%d",ar(1),ar(2)vs),"string")
        start = tic;
        response = read(client, client.NumBytesAvailable, "string");
        while isempty(response)
            response = read(client, client.NumBytesAvailable, "string");
        end
        latency = toc(start);
        latency_vec(n+1) = latency;
    end
    latency_matrix(rep,:)=latency_vec;
end
clear client;
writematrix(latency_matrix,filename);
errorbar(mean(latency_matrix),std(latency_matrix));

