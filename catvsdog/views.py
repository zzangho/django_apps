# -*- coding: utf-8 -*- 
from django.shortcuts import render
from .forms.file_upload import UploadFileForm
from .models import ImageExample
from skimage import io
from .network import Networks as nt
from django.conf import settings


# Create your views here.
def wait_upload(request):
    form = UploadFileForm(request.POST, request.FILES)

    if request.method == 'POST':
        if form.is_valid():
            img = ImageExample(photo= request.FILES['image'] )

            imagearr = io.imread(img.photo)
            imagearr = nt.im_resize_pad(imagearr, 400)

            predict = nt.sess.run(nt.predict_op, feed_dict={nt.X: imagearr})

            if ( predict[0] > predict[1] ):
                confidence = predict[0]
                img.name = '고양이 (확신도:'+str( confidence )+')'
            else:
                confidence = predict[1]
                img.name = '개 (확신도:'+str( confidence )+')'

            if confidence > 0.95:
                img.name='분명히 '+img.name
            elif confidence > 0.6:
                img.name='아마도 '+img.name
            else:
                img.name='애매하지만 '+img.name

#            img.save()

            return render( request, "catvsdog/result.html", {'form': form, 'img':img } )

    return render(request, "catvsdog/wait.html", {'form': form})
