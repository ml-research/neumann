
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
    
    else:
        background_knowledge = """
        """
        return background_knowledge


def get_ilasp_mode_declarations(dataset):
    if dataset in ['twopairs', 'red-triangle']:
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


            #modeh(kp(img)).
            #modeb(2, shape(var(object),const(shape)), (positive)).
            #modeb(2, color(var(object),const(color)), (positive)).
            #modeb(1, diff_color_pair(var(object),var(object)), (positive)).
            #modeb(1, diff_shape_pair(var(object),var(object)), (positive)).
            #modeb(1, closeby(var(object),var(object)), (positive)).
            '''
#modeb(2, same_color_pair(var(object),var(object)), (positive)).
#modeb(2, same_shape_pair(var(object),var(object)), (positive)).

    elif dataset == 'clevr-hans0':
#         mode_declarations = """
# 7 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cube),shape(O1,cylinder),size(O1,large),size(O2,large).
# 7 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cube),shape(O2,sphere),size(O1,small),size(O2,large).
# 7 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),shape(O2,cube),size(O1,large),size(O2,large).
# 7 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),shape(O2,cube),size(O1,small),size(O2,large).
# 6 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),shape(O2,cube),size(O2,large).
# 6 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),shape(O2,cylinder),size(O1,large),size(O2,large).
# 6 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),shape(O2,cylinder),size(O2,large).
# 5 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),shape(O2,sphere).
# 6 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),size(O1,large),size(O2,large).
# 6 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),size(O1,small),size(O2,large).
# 5 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,cylinder),size(O2,large).
# 5 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,sphere),size(O1,small).
# 5 ~ kp(X):-in(O1,X),in(O2,X),shape(O1,sphere),size(O2,small).
# 6 ~ kp(X):-in(O1,X),in(O2,X),shape(O2,sphere),size(O1,small),size(O2,large).
# 5 ~ kp(X):-in(O1,X),in(O2,X),shape(O2,sphere),size(O1,small).
# """
        mode_declarations = """
            #constant(image,img).
            #constant(object,obj0).
            #constant(object,obj1).
            #constant(object,obj2).
            #constant(object,obj3).
            #constant(object,obj4).
            #constant(object,obj5).
            #constant(object,obj6).
            #constant(object,obj7).
            #constant(object,obj8).
            #constant(object,obj9).

            #constant(color,cyan).
            #constant(color,blue).
            #constant(color,yellow).
            #constant(color,purple).
            #constant(color,red).
            #constant(color,green).
            #constant(color,gray).
            #constant(color,brown).
            #constant(shape,sphere).
            #constant(shape,cube).
            #constant(shape,cylinder).
            #constant(material,rubber).
            #constant(material,metal).
            #constant(size,large).
            #constant(size,small).
            #modeh(kp(img)).
            #modeb(2, shape(var(object),const(shape)), (positive)).
            #modeb(2, size(var(object),const(size)), (positive)).
            #modeb(2, color(var(object),const(color)), (positive)).
            """
        
    elif dataset == 'clevr-hans2':
        mode_declarations = """
            #constant(image,img).
            #constant(object,obj0).
            #constant(object,obj1).
            #constant(object,obj2).
            #constant(object,obj3).
            #constant(object,obj4).
            #constant(object,obj5).
            #constant(object,obj6).
            #constant(object,obj7).
            #constant(object,obj8).
            #constant(object,obj9).

            #constant(color,cyan).
            #constant(color,blue).
            #constant(color,yellow).
            #constant(color,purple).
            #constant(color,red).
            #constant(color,green).
            #constant(color,gray).
            #constant(color,brown).
            #constant(shape,sphere).
            #constant(shape,cube).
            #constant(shape,cylinder).
            #constant(material,rubber).
            #constant(material,metal).
            #constant(size,large).
            #constant(size,small).
            #modeh(kp(img)).
            #modeb(2, shape(var(object),const(shape)), (positive)).
            #modeb(2, size(var(object),const(size)), (positive)).
            #modeb(2, color(var(object),const(color)), (positive)).
            """
    return mode_declarations