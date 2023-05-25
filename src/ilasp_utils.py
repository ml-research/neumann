
def get_ilasp_background_knowledge(dataset):
    if dataset == 'twopairs':
        background_knowledge = '''
            diff_color(red,blue).
            diff_color(blue,red).
            diff_color(red,yellow).
            diff_color(yellow,red).
            diff_color(blue,yellow).
            diff_color(yellow,blue).
            diff_shape(circle,square).
            diff_shape(square,circle).
            diff_shape(circle,triangle).
            diff_shape(triangle,circle).
            diff_shape(square,triangle).
            diff_shape(triangle,square).

            same_shape_pair(X,Y):-shape(X,Z),shape(Y,Z).
            same_color_pair(X,Y):-color(X,Z),color(Y,Z).
            diff_shape_pair(X,Y):-shape(X,Z),shape(Y,W),diff_shape(Z,W).
            diff_color_pair(X,Y):-color(X,Z),color(Y,W),diff_color(Z,W).
            '''
        #in4(O1,O2,O3,O4,X):-in(O1,X),in(O2,X),in(O3,X),in(O4,X).
        #P(A):-R1(O1,O2),R2(O1,O2),R3(O3,O4),R4(O3,O4).
        return background_knowledge


def get_ilasp_mode_declarations(dataset):
    if dataset == 'twopairs':
        mode_declarations = '''
            #constant(image,img).
            #constant(object,obj1).
            #constant(object,obj2).
            #constant(object,obj3).
            #constant(object,obj4).
            #constant(object,obj5).
            #constant(object,obj6).

            #constant(color,red).
            #constant(color,blue).
            #constant(color,yellow).
            #constant(shape,circle).
            #constant(shape,square).
            #constant(shape,triangle).

            #modeh(kp(var(image))).
            #modeb(1, color(var(object),const(color))).
            #modeb(1, shape(var(object),const(shape))).
            #modeb(2, same_color_pair(var(object),var(object))).
            #modeb(1, diff_color_pair(var(object),var(object))).
            #modeb(2, same_shape_pair(var(object),var(object))).
            #modeb(1, diff_shape_pair(var(object),var(object))).
            #modeb(1, closeby(var(object),var(object))).
            '''
            #modeb(1, online(var(object),var(object),var(object),var(object),var(object))).
            #modeb(1, in4(var(object),var(object),var(object),var(object),var(image))).
    return mode_declarations