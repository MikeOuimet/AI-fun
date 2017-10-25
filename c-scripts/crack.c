#define _XOPEN_SOURCE 
#include <stdio.h>
#include <string.h>
#include <stdlib.h> 
#include <unistd.h>
// gcc crack.c -lcrypt   to compile




int main(int argc, char **argv)
{
	if (argc != 2) {
		printf("No hashed password given!!\n");
		return 1;
	}

	char *cyph = argv[1];
	printf("Hashed password is: %s\n", cyph);

	char *pwdelem = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_";

	//char *pass = "password";
	//char *hash = crypt(pass, "AB");
	//printf("The hashed is %s\n", hash);

	char salt[3];
	salt[0] = cyph[0]; 
	salt[1] = cyph[1]; 
	salt[2] = '\0';
	//salt[0] = "00\0";
	char testpass[6];
	//testpass =  "aaaaa\0";
	int pwpos = 0;
	char *checkhash;
	for (int p1=0; p1<53; p1++)
	{
		for (int p2=0; p2<53; p2++)
		{
			for (int p3=0; p3<53; p3++)
			{
				for (int p4=0; p4<53; p4++)
				{
					for (int p5=0; p5<53; p5++)
					{
						pwpos = 0;

						if (pwdelem[p1] != '_'){
							testpass[pwpos] = pwdelem[p1];
							pwpos++;
						}

						if (pwdelem[p2] != '_'){
							testpass[pwpos] = pwdelem[p2];
							pwpos++;
						}

						if (pwdelem[p3] != '_'){
							testpass[pwpos] = pwdelem[p3];
							pwpos++;
						}

						if (pwdelem[p4] != '_'){
							testpass[pwpos] = pwdelem[p4];
							pwpos++;
						}

						if (pwdelem[p5] != '_'){
							testpass[pwpos] = pwdelem[p5];
							pwpos++;
						}
						testpass[pwpos] = '\0';
						

					

						checkhash = crypt(testpass, salt);
						//printf("%s\n", crypt(testpass, checkhash));
						if (strcmp(checkhash, cyph) == 0){
							//printf("%i\n", strcmp(checkhash, cyph));
							printf("%s\n", testpass);
							return 0;
						}


					}

				}
			}


		}

	}

	return 0;
}