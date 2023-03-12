#include <stdio.h>
#include <sys/types.h>
#include <signal.h>

int main()
{
	int pid, ret;
	printf("Enter the PID: ");
	scanf("%d", &pid);
	ret = kill(pid,SIGINT);
	printf("ret : %d\n",ret);
}
