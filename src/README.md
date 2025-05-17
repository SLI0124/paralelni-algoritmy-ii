# Zad�n� jednotliv�ch �kol� na cvi�en�

## Cvi�en� 1 - Sezn�men� a znalosti C++

- Alokujte HOST pam�, kter� bude reprezentovat dva M-dimenzion�ln� vektory (A, B), a napl�te je n�jak�mi hodnotami, kde M je dostate�n� velk�. Za�n�te v�ak s mal�m M, abyste b�hem lad�n� vid�li rozumn� v�stupy.
- Alokujte DEVICE pam�, abyste mohli kop�rovat data z HOST.
- Alokujte DEVICE pam� pro ulo�en� v�stupn�ho M-dimenzion�ln�ho vektoru C.
- Vytvo�te kernel, kter� s��t� skal�rn� hodnoty tak, �e $C_i = A_i + B_i$, kde $i\in <0,m-1>$.
- Alokujte HOST pam�, kter� bude reprezentovat 2 M-dimenzion�ln� vektory $A=[a_0,\dots,a_{m-1}]$, $B=[b_0,\dots,b_{m-1}]$, a napl�te je n�jak�mi hodnotami.
- Alokujte DEVICE pam�, abyste mohli kop�rovat data z HOST.
- Alokujte DEVICE pam� pro ulo�en� v�stupn�ch M-dimenzion�ln�ch vektor� $C=[c_0,\dots, c_{m-1}]$.
- Vytvo�te kernel, kter� s��t� vektory tak, �e $C_i = A_i + B_i$.
- ZAMYSLETE SE NAD VARIANTAMI VA�EHO �E�EN�, ZVA�TE KLADY A Z�PORY.

## Cvi�en� 2 - alokace pam�ti, page-locked pam�

- Vytvo�te sloupcovou matici $M[mRows\times mCols]$ obsahuj�c� ��sla 0, 1, 2, 3 $\dots$
- Data by m�la b�t spr�vn� zarovn�na v pam�ti uzam�en� na str�nce (page-locked memory).
- Matice by m�la b�t napln�na v CUDA kernelu.
- Mus�te pou��t Pitch CUDA pam� s odpov�daj�c�m zarovn�n�m.
- Pro zpracov�n� dat mus� b�t pou�ita 2D m��ka 2D blok� o velikosti 8x8.
- Hodnoty matice inkrementujte.
- Nakonec zkop�rujte matici na HOST pomoc� funkce cudaMemcpy2D.

## Cvi�en� 3 - sd�len� pam�

- M�jme jednoduch� ��sticov� syst�m reprezentuj�c� sadu pozic $N$ de��ov�ch kapek v 3D prostoru, kde $N \geq (1<<20)$.
- Vytvo�te vhodnou datovou reprezentaci zm�n�n� sady de��ov�ch kapek.
- M�jme pole 256 v�trn�ch elektr�ren, kter� poskytuj� 256 pohybov�ch vektor�. Tyto pohybov� vektory zp�sobuj� zm�ny pozic v�ech de��ov�ch kapek za jednu sekundu.
- Vytvo�te kernel, kter� simuluje p�d de��ov�ch kapek.
- Pro zjednodu�en� p�edpokl�dejme, �e jedno vol�n� kernelu simuluje jednu sekundu v simulovan�m sv�t�.

## Cvi�en� 4 - konstantn� pam�

- Zkuste napsat jednoduch� k�d, kter� alokuje a nastav� skal�rn� hodnotu v konstantn� pam�ti GPU.
- Zkop�rujte data zp�t na HOST a zkontrolujte hodnotu.

- Ud�lejte tot� s vlastn� strukturou a pot� s n�jak�m polem.

- Na hostiteli vytvo�te pole nazvan� Reference s alespo� $2^{23}$ n�hodn�mi hodnotami s plovouc� desetinnou ��rkou z intervalu $<0, 1>$.
- Vytvo�te pole nazvan� Pattern s alespo� 16 n�hodn�mi hodnotami s plovouc� desetinnou ��rkou z intervalu $<0, 1>$.
- Vytvo�te metodu, kter� najde nejlep�� shodu Pattern v poli Reference. 
- Ot�zky, na kter� mus�te odpov�d�t p�edem:
    - Jak a kde data ulo�it?
    - Jak� je funkce pro porovn�n�?
    - Jak naj�t nejlep�� shodu?
    - Pot�ebuji n�jakou dal�� datovou strukturu?
    - Jak vr�tit v�sledek?

## Cvi�en� 5 - texturov� pam�

TODO

## Cvi�en� 6 - unifikovan� pam� a Texture Object APU

Na�t�te v��kovou mapu a p�ipravte texturu pomoc� CUDA Texture Object API. Vytvo�te norm�lovou mapu z vstupn�ho obr�zku pomoc� Sobelova oper�toru.

## Cvi�en� 7 - Texture reference API vs Object API

OpenGL, freeglut, glew, textury, pixel buffer object

- Postupujte podle pokyn� ze semin��e.
- Pot� se pokuste vy�e�it tyto �koly napsan� v souboru Runner.cu.

## Cvi�en� 8 - Page-locked pam�t a atomick� operace

- Pod�vejte se na alokaci pam�ti v zdrojov�m k�du -> cudaHostAlloc -> pam� uzam�en� na str�nce na hostiteli.
- Najd�te maxim�ln� ��slo v poli s pou�it�m atomick�ch instrukc�.
- Zamyslete se nad �pravami k�du s pou�it�m lok�ln�ho �lo�i�t� dat "per block".
- Zamyslete se nad �pravami k�du s pou�it�m lok�ln�ho �lo�i�t� dat ve sd�len� pam�ti.
- Zamyslete se nad optimalizacemi.
- Zm��te k�d tak, aby pam� uzam�en� na str�nce byla nahrazena unifikovanou pam�t�.
- Zkuste profilovat sv�j k�d v NSight Systems (dom�c� �kol).
- Prove�te detailn� anal�zu kernel� v NSight Compute (dom�c� �kol).

## Cvi�en� 9 - CUDA Streams

- Pokuste se dokon�it danou aplikaci. K tomu mus�te implementovat v�echny d�l�� �koly v k�du. Existuj� dva vektory A a B ($dim \sim= 2^{20}$), kter� budou N-kr�t duplikov�ny v cyklu. Jednoduch� kernel prov�d� sou�et vektor� A+B=C. V�e bude provedeno pomoc� stream� s ohledem na n�sleduj�c� �koly.
    - �KOL 1: Jednoduch� stream
    - �KOL 2: Dva streamy - p��stup hloubky (depth first approach)
    - �KOL 3: Dva streamy - p��stup ���ky (breadth first approach)

## Cvi�en� 10 - CUBLAS knihovna

- Pokuste se dokon�it danou aplikaci. K tomu mus�te implementovat v�echny d�l�� �koly v k�du. Mus�te vytvo�it matici vzd�lenost�, kter� bude obsahovat vzd�lenosti mezi body v prostoru o dimenzi dim.
- Vzd�lenosti mohou b�t vypo��t�ny n�kolika r�zn�mi zp�soby, ale zkuste pou��t funkce BLAS3. To znamen�, �e mus�te pracovat s maticov�mi operacemi.
- Na z�v�r zvy�te po�et bod� a dimenzi.

# Zad�n� jednotliv�ch �kol� za kredit

## Credit task 1 - Zpracov�n� objekt� v n-dimenzion�ln�m prostoru

P�edstavte si, �e se nach�z�te v n-dimenzion�ln�m prostoru, kde je ka�d� objekt pops�n jako n-dimenzion�ln� vektor re�ln�ch ��sel z libovoln�ho rozsahu, ne nutn� z intervalu <0.0; 1.0>. Vygenerujte dostate�n� po�et t�chto objekt� do jedn� velk� matice M0 [m � n], kde m > 2^20 a n je dostate�n� velk� vzhledem k dostupn� pam�ti. Zamyslete se nad orientac� matice (��dkov�/sloupcov�).

1. Diskretizujte zadanou matici podle pravidel, kter� si ur��te, tak aby byl ka�d� objekt/vektor nakonec reprezentov�n vektorem prvk� typu uint8_t nebo uint32_t. Pravidla si m��ete nastavit libovoln�, ale v�echna nastaven� mus� b�t specifikov�na ve form� parametr� metody. Mus� b�t zachov�na topologie dat � tzn. normalizace a dal�� operace mus� b�t provedeny ve stejn�m prostoru a v�echny vektory mus� b�t p�evedeny shodn�. V�stupem bude nov� matice M1 se stejnou orientac� jako M0.
2. Najd�te objekt, kter� je nejvzd�len�j�� od po��tku nov�ho vektorov�ho prostoru (nulov�ho vektoru).
3. Najd�te prvek, kter� je nejvzd�len�j�� od pr�v� nalezen�ho objektu.
4. Pomoc� na�ich pomocn�ch rutin experiment�ln� ov��te dobu b�hu ka�d�ho kernelu.

Maximalizujte po�et operac�, kter� mohou prob�hat paraleln�.

## Credit task 2 - Zad�n� projektu

Projekt je sou��st� va�eho z�v�re�n�ho hodnocen�. Nikdo by nem�l ztr�cet �as a �e�it sv�j projekt a� na konci semestru. Z tohoto d�vodu je tento "kreditn� �kol" zam��en na n�vrh va�eho projektu. M��ete pou��t p�edchoz� popis sv�ho projektu jako z�klad, s t�m rozd�lem, �e budou p�id�ny nezbytn� detaily pro p�esnou implementaci.

�koly: 1. Popi�te kompletn� datovou strukturu va�eho projektu. To znamen�, �e ji� zn�te t�ma projektu a mus�te p�em��let o jeho pot�eb�ch (jak� typ pam�ti, jak budou data ulo�ena, jak budou vl�kna p�istupovat k dat�m atd.). 2. Popi�te algoritmus, kter� bude implementov�n, a jak� bude v�sledek. Popi�te, jak bude algoritmus rozd�len na kernely, metody na za��zen� nebo hostiteli a jak se tyto metody budou navz�jem volat. Mus�te zm�nit v�echny zdroje, kter� chcete pou��t. 3. P�ipravte zpr�vu. Dokument mus� b�t naps�n tak podrobn�, aby jej mohl pou��t jin� �lov�k a p�esn� v�d�l, o �em projekt je a jak bude zpracov�n.

## Credit task 3 - Simulace disease spread

- Postupujte podle pokyn� z posledn�ho semin��e.
- Pot� se pokuste vy�e�it tyto �koly:
- Mapa sv�ta pou��v� pouze n�kolik barev. �ern� p�edstavuje Slovensko a jeho hranice s okoln�mi zem�mi. Zelen� p�edstavuje na�i zemi. B�l� p�edstavuje ostatn� okoln� zem�.
- �erven� bod pobl� m�sta Ko�ice ozna�uje ohnisko slintavky a kulhavky.
- V ka�d�m kroku simulace se �erven� oblast roz�i�uje: ka�d� �ern� pixel, kter� m� alespo� jednoho �erven�ho souseda, se tak� zm�n� na �erven�.
- Ulo�te obr�zek, jakmile �erven� pixel dos�hne hranice s �eskou republikou.

## Credit task 4 - Find pattern in image 

Najd�te pozici pattern0 a pattern1 v p�vodn�m obr�zku (zdrojov�m obr�zku) a vra�te pozici jeho lev�ho horn�ho rohu. Vzory mohou b�t oto�en�!

# Projekt 

Projekt si m��e ka�d� zvolit podle sebe. Jde d�lat ve dvou. S kolegou jsme se domluvili na diferenci�ln� evoluci. 

M�l n�kolik v�tek:

- int r1_i, r2_i, r3_i m�ly b�t unsigned int
- evaluate() by m�lo b�t constexpr, aby se p�i kompilaci v�d�lo, do kter� v�tve m� vkro�it, takhle to v�dy mus� vyhodnocovat v runtime
- v kernelIndividual() po�it�me hranice uvnit�, to se m�lo po��tat venku p�i vol�n� kernelu, takhle to je zbyte�n� overhead, mo�n� je�t� n�co jin�ho
- pr� by je�t� n�jak upravil jednotliv� funkce pro po�it�n� fiteness, pr� by tam dal je�t� v�ce pointr�, bitov�ch operac� a tak, aby to bylo rychlej��, co� fakt nevymysl�m ani za 100 let
