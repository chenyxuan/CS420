#include <bits/stdc++.h>

int main() {
	freopen("cnn_kernel.py","r",stdin);
	freopen("kaggle_kernel.py","w",stdout);
	
	char ch;
	while(scanf("%c", &ch) != EOF) {
		if(ch == '\t') printf("    ");
		else printf("%c", ch);
	}
	fclose(stdin);
	fclose(stdout);
	return 0;
}
