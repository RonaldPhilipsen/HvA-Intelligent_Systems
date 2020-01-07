/* Rules of Rap*/
child(X,Y) :- parent(Y,X).
spouse(X,Y) :- child(P,X), child(P,Y).
husband(X,Y) :- male(X), spouse(X,Y).
wife(X,Y) :- female(X), spouse(X,Y).
son(X,Y) :- male(X), child(X,Y).
daughter(X,Y) :- female(X), child(X,Y).
mother(X,Y) :- female(X), parent(X,Y).
father(X,Y) :- male(X), parent(X,Y).
sibling(X,Y) :- parent(P,X), parent(P,Y), X\=Y.
brother(X,Y) :- male(X), sibling(X,Y).
sister(X,Y) :- female(X), sibling(X,Y).
grandmother(X,Y) :- mother(X,P), parent(P,Y).
grandfather(X,Y) :- father(X,P), parent(P,Y).
great_grandfather(X, Y) :- father(X,P), parent(P, Z), parent(Z,Y). 
grandson(X,Y) :- son(X,P), parent(Y,P).
granddaughter(X,Y) :- daughter(X,P), parent(Y,P).
aunt(X,Y) :- sister(X,P), parent(P,Y).
aunt(X,Y) :- wife(X,P), sibling(P,Q), parent(Q,Y).
uncle(X,Y) :- brother(X,P), parent(P,Y).
uncle(X,Y) :- husband(X,P), sibling(P,Q), parent(Q,Y).
niece(X,Y) :- daughter(X,P), sibling(P,Y).
niece(X,Y) :- daughter(X,P), sibling(P,Q), spouse(Q,Y).
nephew(X,Y) :- son(X,P), sibling(P,Y).
nephew(X,Y) :- son(X,P), sibling(P,Q), spouse(Q,Y).
cousin(X,Y) :- parent(P,X), sibling(P,Q), parent(Q,Y).
ancestor(X,Y):- parent(X,Y).
ancestor(X,Y) :- parent(X,P), ancestor(P,Y).
descendant(X,Y) :- child(X , Y).
descendant(X,Y) :- child(X, Z), descendant(Z, Y).

/* Spitting straight facts yo */
male(ronald).
male(martin).
male(peter).
male(wim).
male(william).
male(tom).
male(hans).
male(marthieu).
male(robert).
male(arthur).
female(sigrit).
female(cor).
female(trees).
female(sarah).
female(marthe).
female(ilse).
female(joyce).
female(evelien).
female(fransje).
female(mandy).
female(zoë).          
parent(peter, ronald).
parent(peter, martin).
parent(sigrit, ronald).
parent(sigrit, martin).
parent(wim, peter).
parent(wim, marthieu).
parent(wim, william).
parent(wim, tom).
parent(wim, hans).
parent(trees, peter).
parent(trees, marthieu).
parent(trees, william).
parent(trees, tom).
parent(trees, hans).
parent(marthieu, sarah).
parent(marthieu, marthe).
parent(fransje, sarah).
parent(fransje, marthe).
parent(hans, robert).
parent(hans, arthur).
parent(william, ilse).
parent(william, joyce).
parent(evelien, ilse).
parent(evelien, joyce).
parent(ilse, zoë).