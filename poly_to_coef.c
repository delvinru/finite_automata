#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int combs_ctr = 1;

void cp(char *dst, char *src, int len) {
    for (int i = 0; i < len; i++)
        dst[i] = src[i];
    dst[len] = '\0';
}

void get_cnk(int rem, int start, int k, char **nums, int n, char **combs, int len) {
    if (rem == 0) {
        for (int i = 0; i < k; i++)
            //printf("%c%c ", combs[0][i*2], combs[0][i*2+1]);
        cp(&combs[combs_ctr][0], &combs[0][0], len*2);
        combs_ctr++;
        //printf("\n");
    } else {
        for (int i = start; i <= n-rem; i++) {
            cp(&combs[0][(k-rem)*2], nums[i], 2);
            get_cnk(rem-1, i+1, k, nums, n, combs, len+1);
        }
    }
}

void get_coeff(int *coeff, char *monom, char len, char **combs, int size, int n) {
    if (monom[0] == '1') {
        coeff[0] = 1;
    } else {
        for (int i = 1; i < size; i++) {
            if (!strcmp(monom, combs[i]))
                coeff[i] = 1;
        }
    }          
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    int size = 2*(int)pow(2, n);

    char **nums = (char **)malloc((n+1)*sizeof(char *));
    for (int i = 0; i < n+1; i++) {
        nums[i] = (char *)malloc(2*sizeof(char));
        nums[i][0] = 'x';
        nums[i][1] = i + '1';
    }

    char **combs = (char **)malloc((size+1)*sizeof(char *));
    for (int i = 0; i < size+1; i++)
        combs[i] = (char *)malloc((n*2+1)*sizeof(char));
    for (int i = 1; i < n+2; i++)
        get_cnk(i, 0, i, nums, n+1, combs, 0);

    int *coeff = (int *)calloc(size, sizeof(int));
   
    char temp = getc(stdin);

    char *monom = (char*)malloc(2*n*sizeof(char));
    char monom_len = 0;

    char x[] = {'x', '\0'};

    while (temp != '\n') {
        if (temp == '+') {
            monom[monom_len]='\0';

            if (!strcmp(monom, x))
                monom[monom_len++] = n + 1 + '0';
            monom[monom_len]='\0';

            get_coeff(coeff, monom, monom_len, combs, size, n);

            memset(monom, 0, monom_len);
            monom_len = 0;
        } else if (temp == ' ') {
        } else {
            monom[monom_len] = temp;
            monom_len++;
        }
        temp = getc(stdin);
    }

    if (monom_len) {
        monom[monom_len]='\0';

        if (!strcmp(monom, x))
            monom[monom_len++] = n + 1 + '0';
        monom[monom_len]='\0';

        get_coeff(coeff, monom, monom_len, combs, size, n); 
    }

    for (int i = 0; i < size; i++)
        printf("%d", coeff[i]);
    printf("\n");
    return 0;
}
